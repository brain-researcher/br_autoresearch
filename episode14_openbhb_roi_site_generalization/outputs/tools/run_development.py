#!/usr/bin/env python3
"""EP14 frozen development comparison; no public-validation input interface.

Run only on a Slurm compute node. The exposed bundle is read-only. Completed
inner-fold losses and outer predictions are reused within an unchanged attempt.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import socket
import time
import warnings

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

SEED = 14014
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
ARMS = ("R0", "R1", "R2", "R3")
CONTROLS = ("random_projection", "permuted_site_projection", "shuffled_age")
EXPECTED_SPLITS = [(2622, 289, 316), (2619, 290, 318), (2615, 292, 320)]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("EP14 computation requires an actual Slurm compute shell")


def bilateral_mapping(names):
    """Same provider ROI and measurement suffix, with lh/rh prefixes only."""
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("duplicate feature names")
    pairs, unpaired, visited = [], [], set()
    for i, name in enumerate(names):
        if i in visited:
            continue
        other = None
        if name.startswith("lh-"):
            other = "rh-" + name[3:]
        elif name.startswith("rh-"):
            other = "lh-" + name[3:]
        if other in index:
            j = index[other]
            left, right = (i, j) if name.startswith("lh-") else (j, i)
            pairs.append((left, right))
            visited.update((i, j))
        else:
            unpaired.append(i)
            visited.add(i)
    if not pairs:
        raise ValueError("no authenticated bilateral names in provider headers")
    return pairs, unpaired


class Scale:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=0)
        self.keep = self.std != 0
        if not self.keep.any():
            raise ValueError("representation collapsed: every fit column constant")
        self.scale = np.where(self.keep, self.std, 1.0)
        return self

    def transform(self, x, discard=False):
        z = (x - self.mean) / self.scale
        return z[:, self.keep] if discard else z


def classifier():
    args = dict(C=1.0, class_weight="balanced", solver="lbfgs",
                max_iter=2000, random_state=SEED)
    # Older sklearn requires explicit multinomial mode; newer releases always
    # use it for multiclass lbfgs and remove this parameter.
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        args["multi_class"] = "multinomial"
    return LogisticRegression(**args)


def fit_classifier(z, labels):
    if len(np.unique(labels)) < 2:
        raise ValueError("site classifier requires at least two fit domains")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = classifier().fit(z, labels)
    return model


def site_directions(z, sites):
    model = fit_classifier(z, sites)
    coef = model.coef_
    # Binary multinomial stores one contrast row. The two centered class rows
    # are its negative and positive, preserving the actual separating span.
    if len(model.classes_) == 2 and coef.shape[0] == 1:
        coef = np.vstack((-coef, coef))
    coef = coef - coef.mean(axis=0)
    _, singular, vt = svd(coef, full_matrices=False)
    tol = max(len(model.classes_), z.shape[1]) * np.finfo(z.dtype).eps * singular[0]
    rank = min(4, int(np.count_nonzero(singular > tol)))
    return vt[:rank], rank


class Representation:
    def __init__(self, arm, names):
        self.arm = "R0" if arm == "shuffled_age" else arm
        self.pairs, self.unpaired = bilateral_mapping(names)
        self.pre = None
        self.removed_rank = 0

    def bilateral(self, x):
        cols = []
        for left, right in self.pairs:
            cols.extend(((x[:, left] + x[:, right]) / 2,
                         x[:, left] - x[:, right]))
        cols.extend(x[:, i] for i in self.unpaired)
        return np.column_stack(cols)

    def fit(self, x, sites):
        if not np.isfinite(x).all():
            raise ValueError("nonfinite fit features")
        if self.arm == "R1":
            z = self.bilateral(x)
        elif self.arm == "R0":
            z = x
        else:
            self.pre = Scale().fit(x)
            z = self.pre.transform(x)
            if self.arm == "R2":
                if min(z.shape) < 64:
                    raise ValueError("PCA64 lacks sufficient dimensions")
                self.pca = PCA(n_components=64, whiten=False, svd_solver="full")
                z = self.pca.fit_transform(z)
                tol = max(x.shape) * np.finfo(z.dtype).eps * self.pca.singular_values_[0]
                if np.count_nonzero(self.pca.singular_values_ > tol) != 64:
                    raise ValueError("PCA64 has deficient effective rank")
            else:
                if self.arm == "permuted_site_projection":
                    labels = np.random.RandomState(SEED).permutation(sites)
                    self.directions, self.removed_rank = site_directions(z, labels)
                else:
                    self.directions, self.removed_rank = site_directions(z, sites)
                    if self.arm == "random_projection":
                        random = np.random.RandomState(SEED).normal(
                            size=(z.shape[1], self.removed_rank))
                        self.directions = np.linalg.qr(random, mode="reduced")[0].T
                    elif self.arm != "R3":
                        raise ValueError(self.arm)
                z = z - (z @ self.directions.T) @ self.directions
        self.final = Scale().fit(z)
        transformed = self.final.transform(z, discard=True)
        if not np.isfinite(transformed).all():
            raise ValueError("nonfinite final fit representation")
        self.n_fit = len(x)
        self.n_input_features = x.shape[1]
        return transformed

    def transform(self, x):
        """Features only: this method cannot receive age, site or role labels."""
        if self.arm == "R1":
            z = self.bilateral(x)
        elif self.arm == "R0":
            z = x
        else:
            z = self.pre.transform(x)
            if self.arm == "R2":
                z = self.pca.transform(z)
            else:
                z = z - (z @ self.directions.T) @ self.directions
        z = self.final.transform(z, discard=True)
        if not np.isfinite(z).all():
            raise ValueError("nonfinite held-role representation")
        return z


def ridge_grid(z_fit, y_fit, z_test, alphas=ALPHAS):
    """Exact centered ridge with unpenalized intercept; reuse one full SVD."""
    center, intercept = z_fit.mean(axis=0), y_fit.mean()
    u, singular, vt = svd(z_fit - center, full_matrices=False)
    tolerance = max(z_fit.shape) * np.finfo(z_fit.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        raise ValueError("ridge input collapsed")
    projected = u.T @ (y_fit - intercept)
    prediction = []
    for alpha in alphas:
        weights = vt.T @ (singular / (singular**2 + alpha) * projected)
        prediction.append((z_test - center) @ weights + intercept)
    return np.column_stack(prediction), rank


def validate_rank(rep, rank):
    if rep.arm == "R2" and rank != 64:
        raise ValueError("final PCA representation rank does not equal 64")
    if rep.arm in ("R3", "random_projection", "permuted_site_projection"):
        if rank > min(rep.n_fit - 1, rep.n_input_features - rep.removed_rank):
            raise ValueError("projected representation retains a removed direction")


class Bundle:
    def __init__(self, root):
        root = Path(root)
        if root.name != "exposed":
            raise ValueError("input root must be the exposed development branch")
        self.root = root
        self.ids = np.load(root / "train_row_ids.npy", allow_pickle=False).astype(str)
        if len(self.ids) != 3227 or len(np.unique(self.ids)) != 3227:
            raise ValueError("development IDs differ from 3227 unique participants")
        with (root / "train_targets.csv").open() as f:
            reader = csv.DictReader(f)
            self.target_headers = reader.fieldnames
            rows = list(reader)
        if len(rows) != 3227 or not {"age", "siteXacq"}.issubset(self.target_headers):
            raise ValueError("development target shape or headers do not match")
        candidates = [k for k in self.target_headers
                      if all(str(row[k]) == self.ids[i] for i, row in enumerate(rows))]
        if len(candidates) != 1:
            raise ValueError("targets require one exact ordered opaque-ID join")
        self.join_column = candidates[0]
        self.age = np.array([float(row["age"]) for row in rows])
        self.sites = np.array([str(row["siteXacq"]) for row in rows])
        if not np.isfinite(self.age).all() or any(s in ("", "nan", "None") for s in self.sites):
            raise ValueError("missing development targets")
        self.splits = []
        raw = read_json(root / "development_splits.json")["folds"]
        lookup = {key: i for i, key in enumerate(self.ids)}
        for view in range(3):
            fold = raw[f"fold{view}"]
            fit_keys = [key for key in ("train", "fit", "training") if key in fold]
            if len(fit_keys) != 1:
                raise ValueError(f"unknown fit key: {list(fold)}")
            split = {role: np.array([lookup[str(x)] for x in fold[key]])
                     for role, key in (("fit", fit_keys[0]), ("internal", "internal_test"),
                                       ("external", "external_test"))}
            if tuple(len(split[k]) for k in ("fit", "internal", "external")) != EXPECTED_SPLITS[view]:
                raise ValueError(f"split {view} role counts differ from contract")
            joined = np.concatenate(list(split.values()))
            if len(joined) != 3227 or len(np.unique(joined)) != 3227:
                raise ValueError("split role overlap or missing participant")
            if set(self.sites[split["external"]]) & set(self.sites[split["fit"]]):
                raise ValueError("external domains overlap fit domains")
            if not set(self.sites[split["internal"]]) <= set(self.sites[split["fit"]]):
                raise ValueError("internal domains are unseen")
            self.splits.append(split)

    def atlas(self, atlas):
        names = read_json(self.root / f"feature_names_{atlas}.json")
        x = np.load(self.root / f"train_{atlas}_X.npy", allow_pickle=False)
        expected = 476 if atlas == "desikan" else 1036
        if x.shape != (3227, expected) or len(names) != expected:
            raise ValueError("atlas shape or feature count differs from contract")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite development ROI value")
        bilateral_mapping(names)
        return np.asarray(x, dtype=np.float64), names


def validity_regressions():
    """Contract-required leakage and estimator checks, without holdout data."""
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(SEED)
    fit = rng.normal(size=(120, 70))
    held = rng.normal(size=(20, 70)) + 8
    age = rng.normal(size=120)
    sites = np.tile(np.arange(6).astype(str), 20)
    names = [f"{hemi}-region{i}_thickness" for hemi in ("lh", "rh") for i in range(35)]
    checks = {}
    for arm in ARMS:
        rep = Representation(arm, names)
        z = rep.fit(fit, sites)
        before = rep.final.mean.copy()
        zh = rep.transform(held)
        rep.transform(held * 100)
        assert np.array_equal(before, rep.final.mean), "transform changed fit state"
        pred, rank = ridge_grid(z, age, zh)
        validate_rank(rep, rank)
        for j, alpha in enumerate(ALPHAS):
            actual = Ridge(alpha=alpha, solver="svd").fit(z, age).predict(zh)
            np.testing.assert_allclose(pred[:, j], actual, rtol=1e-8, atol=1e-8)
        assert rank > 0 and rep.n_fit == len(fit)
        checks[arm] = {"ridge_matches_sklearn": True, "transform_state_unchanged": True,
                       "rank": rank, "removed_rank": rep.removed_rank}
    expected = fit.mean(axis=0)
    good = Scale().fit(fit)
    bad = Scale().fit(np.vstack((fit, held)))
    np.testing.assert_allclose(good.mean, expected)
    if np.allclose(bad.mean, expected):
        raise AssertionError("deliberately leaky preprocessing was not detected")
    if list(inspect.signature(Representation.transform).parameters) != ["self", "x"]:
        raise AssertionError("held-role transform accepts labels")
    checks["deliberately_leaky_preprocessing_detected"] = True
    return checks


def prepare(bundle, out):
    report = {"analysis_id": "prepared_bundle_check", "rows": len(bundle.ids),
              "target_join": bundle.join_column, "atlases": {}, "splits": [],
              "input_root": str(bundle.root), "holdout_access": False}
    for atlas in ("desikan", "destrieux"):
        x, names = bundle.atlas(atlas)
        pairs, unpaired = bilateral_mapping(names)
        report["atlases"][atlas] = {"shape": list(x.shape), "finite": True,
                                     "bilateral_pairs": len(pairs), "unpaired": len(unpaired)}
        write_json(out / f"{atlas}_bilateral_mapping.json",
                   {"feature_names": names, "pairs_left_right": pairs, "unpaired": unpaired})
    for split in bundle.splits:
        report["splits"].append({k: {"rows": len(v), "domains": len(np.unique(bundle.sites[v]))}
                                  for k, v in split.items()})
    report["validity_regressions"] = validity_regressions()
    write_json(out / "prepared_bundle_check.json", report)
    return report


def inner_fold(bundle, x, names, atlas, arm, view, domain_number, out):
    cache = out / "inner" / f"{atlas}_{arm}_view{view}_domain{domain_number}.json"
    if cache.exists():
        return read_json(cache)
    start = time.monotonic()
    fit = bundle.splits[view]["fit"]
    domains = np.unique(bundle.sites[fit])
    domain = domains[domain_number]
    inner_train = fit[bundle.sites[fit] != domain]
    inner_valid = fit[bundle.sites[fit] == domain]
    y = bundle.age.copy()
    if arm == "shuffled_age":
        # Permute within this inner fit role, so validation ages cannot become
        # training labels through a permutation over the whole outer fit set.
        y[inner_train] = np.random.RandomState(SEED).permutation(y[inner_train])
    rep = Representation(arm, names)
    z = rep.fit(x[inner_train], bundle.sites[inner_train])
    prediction, rank = ridge_grid(z, y[inner_train], rep.transform(x[inner_valid]))
    validate_rank(rep, rank)
    # Inner validation ages remain untouched, including for the null control.
    losses = np.mean(np.abs(prediction - y[inner_valid, None]), axis=0)
    record = {"atlas": atlas, "arm": arm, "view": view, "domain": str(domain),
              "mae_by_alpha": losses.tolist(), "fit_rows": len(inner_train),
              "validation_rows": len(inner_valid), "effective_rank": rank,
              "removed_rank": rep.removed_rank, "seconds": time.monotonic() - start}
    write_json(cache, record)
    print(json.dumps({"stage": "inner", **record}), flush=True)
    return record


def outer_fit(bundle, x, names, atlas, arm, view, out):
    cache = out / "outer" / f"{atlas}_{arm}_view{view}.json"
    if cache.exists():
        return read_json(cache)
    start = time.monotonic()
    split = bundle.splits[view]
    n_domains = len(np.unique(bundle.sites[split["fit"]]))
    inner = [inner_fold(bundle, x, names, atlas, arm, view, k, out) for k in range(n_domains)]
    means = np.mean([r["mae_by_alpha"] for r in inner], axis=0)
    selected_index = min(range(len(ALPHAS)), key=lambda i: (means[i], -ALPHAS[i]))
    alpha = float(ALPHAS[selected_index])
    fit = split["fit"]
    y_fit = bundle.age[fit]
    if arm == "shuffled_age":
        y_fit = np.random.RandomState(SEED).permutation(y_fit)
    rep = Representation(arm, names)
    z = rep.fit(x[fit], bundle.sites[fit])
    both = np.concatenate((split["internal"], split["external"]))
    held = rep.transform(x[both])
    prediction, rank = ridge_grid(z, y_fit, held, np.array([alpha]))
    validate_rank(rep, rank)
    probe = fit_classifier(z, bundle.sites[fit])
    n_internal = len(split["internal"])
    site_pred = probe.predict(held[:n_internal])
    appearances = []
    for i, row in enumerate(both):
        internal = i < n_internal
        record = {"id": str(bundle.ids[row]), "domain": str(bundle.sites[row]),
                  "age": float(bundle.age[row]), "prediction": float(prediction[i, 0]),
                  "absolute_error": float(abs(prediction[i, 0] - bundle.age[row])),
                  "median_error": float(abs(np.median(bundle.age[fit]) - bundle.age[row])),
                  "role": "internal" if internal else "external", "view": view}
        if internal:
            record["site_correct"] = int(site_pred[i] == bundle.sites[row])
        appearances.append(record)
    record = {"atlas": atlas, "arm": arm, "view": view, "alpha": alpha,
              "inner_domain_macro_mae": means.tolist(), "inner_domains": n_domains,
              "final_dimension": z.shape[1], "effective_rank": rank,
              "removed_rank": rep.removed_rank, "appearances": appearances,
              "seconds": time.monotonic() - start}
    write_json(cache, record)
    print(json.dumps({k: v for k, v in record.items() if k != "appearances"}), flush=True)
    return record


def collapse(records, role, field):
    by_id = {}
    for outer in records:
        for row in outer["appearances"]:
            if row["role"] == role:
                item = by_id.setdefault(row["id"], {"domain": row["domain"], "age": row["age"], "values": []})
                if item["domain"] != row["domain"] or item["age"] != row["age"]:
                    raise ValueError("participant metadata differs between views")
                item["values"].append(row[field])
    return {key: {"domain": v["domain"], "age": v["age"], "loss": float(np.mean(v["values"]))}
            for key, v in by_id.items()}


def summarize(collapsed):
    domains = sorted({v["domain"] for v in collapsed.values()})
    by_domain = {}
    for domain in domains:
        values = [v["loss"] for v in collapsed.values() if v["domain"] == domain]
        by_domain[domain] = {"participants": len(values), "mean": float(np.mean(values))}
    domain_means = [by_domain[d]["mean"] for d in domains]
    quartiles = np.quantile([v["age"] for v in collapsed.values()], [0.25, 0.5, 0.75])
    age_errors = []
    for q in range(4):
        values = [v["loss"] for v in collapsed.values() if np.searchsorted(quartiles, v["age"], side="right") == q]
        age_errors.append({"quartile": q + 1, "participants": len(values),
                           "mean": float(np.mean(values)) if values else None})
    return {"domain_macro": float(np.mean(domain_means)), "worst_domain": float(np.max(domain_means)),
            "participant_pooled": float(np.mean([v["loss"] for v in collapsed.values()])),
            "domains": by_domain, "age_quartile_edges": quartiles.tolist(), "age_quartiles": age_errors,
            "leave_one_domain_out": {d: float(np.mean([by_domain[k]["mean"] for k in domains if k != d]))
                                      for d in domains} if len(domains) > 1 else {}}


def paired_interval(base, other, relative=False):
    if set(base) != set(other):
        raise ValueError("paired interval participant mismatch")
    domains = sorted({v["domain"] for v in base.values()})
    groups = []
    for d in domains:
        keys = sorted(k for k, v in base.items() if v["domain"] == d)
        groups.append(np.array([[base[k]["loss"], other[k]["loss"]] for k in keys]))
    rng = np.random.RandomState(SEED)
    values = np.empty(10000)
    for b in range(10000):
        means = []
        for d in rng.randint(len(groups), size=len(groups)):
            group = groups[d]
            means.append(group[rng.randint(len(group), size=len(group))].mean(axis=0))
        ref, alt = np.mean(means, axis=0)
        values[b] = (alt - ref) / ref if relative else alt - ref
    return {"upper_one_sided_95": float(np.quantile(values, 0.95)),
            "descriptive_percentiles_2_5_97_5": np.quantile(values, [0.025, 0.975]).tolist(),
            "replicates": 10000, "seed": SEED, "scope": "observed_domains_only_descriptive_development"}


def aggregate(out):
    tables = {}
    all_groups = {"desikan": ARMS + CONTROLS, "destrieux": ARMS}
    for atlas, arms in all_groups.items():
        tables[atlas] = {}
        for arm in arms:
            records = [read_json(out / "outer" / f"{atlas}_{arm}_view{v}.json") for v in range(3)]
            age = collapse(records, "external", "absolute_error")
            site = collapse(records, "internal", "site_correct")
            tables[atlas][arm] = {"external_age": summarize(age), "internal_site": summarize(site),
                                  "internal_age": summarize(collapse(records, "internal", "absolute_error")),
                                  "age_median": summarize(collapse(records, "external", "median_error")),
                                  "per_view_external": [summarize(collapse([r], "external", "absolute_error")) for r in records],
                                  "per_view_internal_site": [summarize(collapse([r], "internal", "site_correct")) for r in records],
                                  "final_dimensions": [r["final_dimension"] for r in records],
                                  "effective_ranks": [r["effective_rank"] for r in records],
                                  "removed_ranks": [r["removed_rank"] for r in records],
                                  "alphas": [r["alpha"] for r in records]}
            if arm == "R0":
                base_age, base_site = age, site
            else:
                base = tables[atlas]["R0"]
                current = tables[atlas][arm]
                delta = current["external_age"]["domain_macro"] - base["external_age"]["domain_macro"]
                current["delta_age"] = delta
                current["relative_age_change"] = delta / base["external_age"]["domain_macro"]
                current["delta_site"] = current["internal_site"]["domain_macro"] - base["internal_site"]["domain_macro"]
                current["paired_domain_age_effects"] = {
                    d: current["external_age"]["domains"][d]["mean"] - base["external_age"]["domains"][d]["mean"]
                    for d in base["external_age"]["domains"]}
                current["paired_domain_site_effects"] = {
                    d: current["internal_site"]["domains"][d]["mean"] - base["internal_site"]["domains"][d]["mean"]
                    for d in base["internal_site"]["domains"]}
                current["leave_one_domain_out_age_change"] = {
                    d: current["external_age"]["leave_one_domain_out"][d] - value
                    for d, value in base["external_age"]["leave_one_domain_out"].items()}
                current["leave_one_domain_out_site_change"] = {
                    d: current["internal_site"]["leave_one_domain_out"][d] - value
                    for d, value in base["internal_site"]["leave_one_domain_out"].items()}
                current["per_view_paired_effects"] = [
                    {"view": v,
                     "delta_age": current["per_view_external"][v]["domain_macro"] - base["per_view_external"][v]["domain_macro"],
                     "delta_site": current["per_view_internal_site"][v]["domain_macro"] - base["per_view_internal_site"][v]["domain_macro"]}
                    for v in range(3)]
                current["relative_age_interval"] = paired_interval(base_age, age, relative=True)
                current["absolute_age_interval"] = paired_interval(base_age, age)
                current["site_interval"] = paired_interval(base_site, site)
                current["eligible"] = arm in ARMS and current["relative_age_change"] <= -0.02 and current["delta_site"] <= 0.01
    eligible = [a for a in ARMS[1:] if tables["desikan"][a]["eligible"]]
    # This last deterministic order is used only after all exact scientific ties.
    simplicity = {"R1": 0, "R2": 1, "R3": 2}
    selected = min(eligible, key=lambda a: (tables["desikan"][a]["external_age"]["domain_macro"],
                                           tables["desikan"][a]["external_age"]["worst_domain"],
                                           tables["desikan"][a]["internal_site"]["domain_macro"],
                                           simplicity[a])) if eligible else None
    report = {"analysis_id": "fixed_development_comparison", "results": tables["desikan"],
              "selected_candidate": selected, "selection_is_confirmation": False,
              "holdout_access": False,
              "disposition": "candidate_selected_lock_and_exposure_review_pending" if selected else "development_no_candidate"}
    write_json(out / "development_report.json", report)
    write_json(out / "controls_report.json", {"analysis_id": "controls_and_sensitivity",
               "desikan_falsifiers": {a: tables["desikan"][a] for a in CONTROLS},
               "destrieux_sensitivity": tables["destrieux"], "changes_primary_selection": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "representative", "run", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=Path("inputs/openbhb/exposed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    compute_only()
    if args.threads < 1 or args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("thread request exceeds allocation")
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        bundle = Bundle(args.data_root)
        if args.stage == "prepare":
            result = prepare(bundle, args.out)
        elif args.stage == "representative":
            if not (args.out / "prepared_bundle_check.json").exists():
                prepare(bundle, args.out)
            x, names = bundle.atlas("destrieux")
            result = inner_fold(bundle, x, names, "destrieux", "R3", 0, 0, args.out)
            write_json(args.out / "representative_cost.json", result)
        elif args.stage == "run":
            if not (args.out / "prepared_bundle_check.json").exists():
                raise ValueError("run requires completed prepared_bundle_check")
            for atlas, arms in (("desikan", ARMS + CONTROLS), ("destrieux", ARMS)):
                x, names = bundle.atlas(atlas)
                for arm in arms:
                    for view in range(3):
                        outer_fit(bundle, x, names, atlas, arm, view, args.out)
            result = aggregate(args.out)
        else:
            result = aggregate(args.out)
        print(json.dumps({"stage": args.stage, "complete": True,
                          "selected_candidate": result.get("selected_candidate")}), flush=True)


if __name__ == "__main__":
    main()
