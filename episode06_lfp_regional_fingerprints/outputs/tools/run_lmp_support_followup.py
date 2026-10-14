"""One fixed, exposed-development EP06 M1 LMP support analysis.

All fits/predictions are published before evaluation outcomes are constructed.
No calibration values, reserved session or source MAT reader is used. Existing
all-unit LMP predictions are reused. New targets deliberately change population
support; their contrasts cannot identify contamination as the cause of a change.
"""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import pickle
import resource
import time

import numpy as np

import bandwise_prediction as bp
from cca_mapping import fit_cca_mapping, predict_cca_mapping
from component_scoring import BoundActivity
from development_views import CandidateDevelopmentViews, DevelopmentOutcomeViews
from elapsed_time_baseline import fit_elapsed_time_baseline, predict_elapsed_time_baseline
import run_initial_profiles as initial


ANALYSIS = initial.EPISODE / "outputs/runs/lmp_support_followup"
SESSIONS = initial.SESSIONS
FAMILY = "starting_09"
RECIPE = dict(initial.FAMILIES)[FAMILY]
DRAWS = 3
MISMATCHES = 199
METHODS = ("lmp", "high_frequency", "lmp_after_hf_projection")


class Unavailable(ValueError):
    """An explicit unavailable scientific cell; never a replacement fit."""


def program():
    return {"analysis": "lmp_support_followup", "version": 1,
        "sessions": [s.filename for s in SESSIONS], "region": "M1",
        "support_draws": DRAWS, "mismatch_draws": MISMATCHES,
        "methods": list(METHODS), "rank": 2, "cca_regularization": .01,
        "prediction_penalty": .01, "hf_projection_penalty": .01,
        "hf_bands": ["100_200", "200_400"], "calibration_values_used": False,
        "reserved_sessions_opened": False, "original_search_remains_closed": True,
        "inference": "descriptive_exposed_development_no_selection_adjusted_pvalues"}


def support_plan(metadata):
    """Freeze all seven support choices from physical IDs, before neural values."""
    pools = {}
    for spec in SESSIONS:
        meta = metadata[spec.filename]
        selected = set(meta.selected_electrode_ids)
        guide = meta.native_unit_guide
        near = tuple(i for i, row in enumerate(guide) if row[0] in selected)
        disjoint = tuple(i for i, row in enumerate(guide) if row[0] not in selected)
        if set(near) & set(disjoint) or sorted(near + disjoint) != list(range(len(guide))):
            raise ValueError("unit guides must partition exactly by selected physical electrodes")
        pools[spec.filename] = {"on_electrodes": near, "disjoint_electrodes": disjoint}
    count = min(len(pool) for cells in pools.values() for pool in cells.values())
    plans = {}
    for si, spec in enumerate(SESSIONS):
        meta = metadata[spec.filename]
        plans[spec.filename] = [{"id": "all_units", "support": "all_units", "draw": None,
            "indices": list(range(len(meta.native_unit_guide)))}]
        if count >= 2:
            for support_index, (name, pool) in enumerate(pools[spec.filename].items()):
                for draw in range(DRAWS):
                    rng = np.random.default_rng(np.random.SeedSequence([6006301, si, support_index, draw]))
                    indices = sorted(int(i) for i in rng.choice(pool, count, replace=False))
                    plans[spec.filename].append({"id": f"{name}_{draw}", "support": name,
                        "draw": draw, "indices": indices})
    return {"matched_unit_count": count, "matched_support_available": count >= 2,
        "unavailable_reason": None if count >= 2 else "fewer_than_two_units_in_a_required_pool",
        "pool_counts": {s: {k: len(v) for k, v in p.items()} for s, p in pools.items()},
        "plans": plans}


def restrict(values, indices):
    design = replace(values.design, axis_ids=tuple(values.design.axis_ids[i] for i in indices))
    return BoundActivity(design, bp._readonly(values.values[:, :, indices]))


def context_for(original, counts):
    """Refit only new support's full-training axes/template; original stays exact."""
    if counts.design == original.training_design:
        return original
    n, bins, units = counts.values.shape
    if units < 2:
        raise Unavailable("fewer_than_two_units")
    try:
        mean, loadings, singular, rank, tol = bp._pca(counts.values.reshape(n*bins, units), 2, "population")
    except ValueError as error:
        if "exceeds numerical training rank" in str(error):
            raise Unavailable(str(error)) from error
        raise
    template = np.stack([counts.values[rows].mean(axis=0) for rows in bp._direction_rows(counts.design)])
    return replace(original, training_design=counts.design, population_mean=bp._readonly(mean),
        loadings=bp._readonly(loadings), singular_values=bp._readonly(singular),
        numerical_rank=rank, rank_tolerance=tol, training_template=bp._readonly(template))


def fit_map(x, y, check):
    """Same electrode PCA2 and rank-one CCA as original, for an explicit matrix."""
    check()
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(x) != len(y):
        raise ValueError("complete finite feature/target rows required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    standardized = (x - mean) / scale
    try:
        pca_mean, axes, _, _, _ = bp._pca(standardized, 2, "electrode")
    except ValueError as error:
        if "exceeds numerical training rank" in str(error):
            raise Unavailable(str(error)) from error
        raise
    mapped = fit_cca_mapping((standardized-pca_mean) @ axes, y,
        covariance_regularization=.01, prediction_penalty=.01, check=check)
    return {"mean": mean, "scale": scale, "pca_mean": pca_mean, "axes": axes, "map": mapped}


def predict_map(fitted, x, check):
    scores = ((x-fitted["mean"])/fitted["scale"]-fitted["pca_mean"]) @ fitted["axes"]
    return predict_cca_mapping(fitted["map"], scores, check=check)


def remove_hf(training_lmp, training_hf, evaluation_lmp, evaluation_hf):
    """Training-only ridge projection; no claim of perfect orthogonalization."""
    mean, scale = training_hf.mean(axis=0), training_hf.std(axis=0)
    scale[scale == 0] = 1.
    x = (training_hf-mean)/scale
    ymean = training_lmp.mean(axis=0)
    n = len(x)
    a = np.vstack((x/math.sqrt(n), math.sqrt(.01)*np.eye(x.shape[1])))
    b = np.vstack(((training_lmp-ymean)/math.sqrt(n), np.zeros((x.shape[1], training_lmp.shape[1]))))
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    state = {"mean": mean, "scale": scale, "intercept": ymean, "coefficients": coefficients}
    return training_lmp-(x@coefficients+ymean), evaluation_lmp-((evaluation_hf-mean)/scale@coefficients+ymean), state


def half_rows(design):
    halves = []
    for half in (0, 1):
        trials = sorted(i for rows in bp._direction_rows(design) for i in rows[half::2])
        if any(sum(design.trial_directions[i] == d for i in trials) < 2 for d in design.directions):
            raise Unavailable("training_half_has_fewer_than_two_trials_per_direction")
        halves.append(np.asarray([i*len(design.time_ids)+b for i in trials for b in range(len(design.time_ids))]))
    return halves


def correlation(left, right):
    left, right = np.ravel(left), np.ravel(right)
    left, right = left-left.mean(), right-right.mean()
    denominator = float(np.linalg.norm(left)*np.linalg.norm(right))
    return None if denominator == 0 else float(left@right/denominator)


def mismatch_indices(design, seed, number=MISMATCHES):
    """Fixed random-cycle derangements; descriptive reference, not uniform null."""
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(number):
        mapping = np.arange(len(design.trial_ids))
        for rows in bp._direction_rows(design):
            if len(rows) < 2:
                raise Unavailable("no_within_direction_derangement")
            ordered = rng.permutation(rows)
            mapping[ordered] = np.roll(ordered, int(rng.integers(1, len(rows))))
        result.append(mapping)
    return result


def score_array(target, prediction):
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("matching finite complete targets/predictions required")
    energy = float(np.sum(target*target))
    if not math.isfinite(energy) or energy <= 0:
        raise Unavailable("zero_or_nonfinite_residual_energy")
    return float(1.-np.sum((target-prediction)**2)/energy)


def score_cell(cell, counts, mappings, check):
    if "unavailable" in cell:
        return {k: cell[k] for k in ("session", "animal", "plan", "unavailable")}
    context = cell["context"]
    observed = restrict(counts, cell["plan"]["indices"])
    target = bp.transform_observations(context, observed, region="M1", check=check).values
    result = {k: cell[k] for k in ("session", "animal", "plan", "training_diagnostics", "reliability")}
    result["methods"] = {}
    result["residual_energy"] = float(np.sum(target**2))
    try:
        result["elapsed_score"] = score_array(target, cell["elapsed_prediction"])
        for name, prediction in cell["predictions"].items():
            check()
            if prediction is None:
                result["methods"][name] = {"unavailable": cell["method_unavailable"][name]}
                continue
            value = score_array(target, prediction)
            mismatched = []
            for mapping in mappings:
                check()
                mismatched.append(score_array(target, prediction[mapping]))
            center = float(np.mean(mismatched))
            result["methods"][name] = {"score": value,
                "minus_elapsed": value-result["elapsed_score"],
                "mismatch_mean": center, "minus_mismatch": value-center,
                "mismatch_scores": mismatched}
        result["paired_changes"] = {}
        for left, right in (("lmp_after_hf_projection", "lmp"), ("lmp", "high_frequency")):
            a, b = result["methods"][left], result["methods"][right]
            result["paired_changes"][f"{left}_minus_{right}"] = (
                None if "unavailable" in a or "unavailable" in b else a["score"]-b["score"])
        result["reliability"]["half_scores"] = [None if value is None else score_array(target, value)
            for value in cell["half_predictions"]]
    except Unavailable as error:
        result["unavailable"] = str(error)
    return result


def summary(rows):
    """Average subset draws, then dates within animal, then the two animals."""
    output = {}
    for support in ("all_units", "on_electrodes", "disjoint_electrodes"):
        for method in METHODS:
            key = f"{support}/{method}"
            dates = []
            for spec in SESSIONS:
                cells = [row for row in rows if row["session"] == spec.filename and row["plan"]["support"] == support]
                expected = 1 if support == "all_units" else DRAWS
                complete = len(cells) == expected and all("unavailable" not in row and
                    "unavailable" not in row["methods"][method] for row in cells)
                values = None if not complete else {metric: float(np.mean([row["methods"][method][metric] for row in cells]))
                    for metric in ("score", "minus_elapsed", "minus_mismatch")}
                dates.append({"session": spec.filename, "animal": spec.animal, "values": values})
            if any(row["values"] is None for row in dates):
                output[key] = {"complete": False, "dates": dates, "animal_means": None, "equal_animal_mean": None}
                continue
            animals = {a: {metric: float(np.mean([r["values"][metric] for r in dates if r["animal"] == a]))
                for metric in ("score", "minus_elapsed", "minus_mismatch")} for a in ("Mihili", "Chewie-L")}
            output[key] = {"complete": True, "dates": dates, "animal_means": animals,
                "equal_animal_mean": {m: float(np.mean([a[m] for a in animals.values()])) for m in animals["Mihili"]},
                "omission_equal_animal_means": {r["session"]: {m: float(np.mean([np.mean([v["values"][m] for v in dates
                    if v["animal"] == a and v["session"] != r["session"]]) for a in animals]))
                    for m in animals["Mihili"]} for r in dates}}
    return output


def original_cells(views_path, check):
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as stream:
        if json.load(stream) != {"program": initial._program(), "views_path": str(views_path)}:
            raise ValueError("use exact original completed development views")
    with initial._owned_read(initial.ATTEMPT_PATH / "fit_complete.json") as stream:
        if json.load(stream).get("completed_cells") != 84:
            raise ValueError("original fits are incomplete")
    cells = {}
    with initial._owned_read(initial.ATTEMPT_PATH / f"{FAMILY}.pickle", binary=True) as stream:
        for spec in SESSIONS:
            for region in initial.REGIONS:
                cell = pickle.load(stream)
                initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                if region == "M1":
                    cells[spec.filename] = cell
        if stream.read(1):
            raise ValueError("unexpected original cells")
    return cells


def fit_cell(spec, plan, original, train, features, train_meta, eval_meta, check):
    cell = {"session": spec.filename, "animal": spec.animal, "plan": plan}
    counts = restrict(train[1], plan["indices"])
    try:
        context = context_for(original["fit"], counts)
        target = bp.transform_observations(context, counts, region="M1", check=check).values.reshape(-1, 2)
        lmp = train[0].values[:, :, :, 0].reshape(-1, 15)
        held = features.values[:, :, :, 0].reshape(-1, 15)
        hf = train[0].values[:, :, :, 7:9].reshape(-1, 30)
        held_hf = features.values[:, :, :, 7:9].reshape(-1, 30)
        clean, held_clean, projection = remove_hf(lmp, hf, held, held_hf)
        shape = (len(features.design.trial_ids), len(features.design.time_ids), 2)
        predictions, maps, unavailable = {}, {}, {}
        for method, x, held_x in zip(METHODS, (lmp, hf, clean), (held, held_hf, held_clean)):
            check()
            if method == "lmp" and plan["id"] == "all_units":
                predictions[method] = original["predictions"].predictions[0].values
                maps[method] = "reused_original_starting09_lmp"
            else:
                try:
                    maps[method] = fit_map(x, target, check)
                    predictions[method] = predict_map(maps[method], held_x, check).reshape(shape)
                except Unavailable as error:
                    predictions[method] = None
                    unavailable[method] = str(error)
        halves, half_states, half_reasons = [], [], []
        for rows in half_rows(counts.design):
            try:
                fitted = fit_map(lmp[rows], target[rows], check)
                halves.append(predict_map(fitted, held, check).reshape(shape))
                half_states.append(fitted)
                half_reasons.append(None)
            except Unavailable as error:
                halves.append(None)
                half_states.append(None)
                half_reasons.append(str(error))
        tm = replace(train_meta, design=counts.design,
            native_unit_guide=tuple(train_meta.native_unit_guide[i] for i in plan["indices"]))
        em = replace(eval_meta, design=replace(eval_meta.design, axis_ids=counts.design.axis_ids),
            native_unit_guide=tm.native_unit_guide)
        timing = fit_elapsed_time_baseline(context, tm, counts, check=check)
        elapsed = predict_elapsed_time_baseline(timing, em, check=check).values
        reliability = {"prediction_correlation": None if any(v is None for v in halves) else correlation(*halves),
            "half_unavailable": half_reasons,
            "scope": "mapping_stability_conditional_on_shared_full_training_population_axes_and_template"}
        cell.update(context=context, predictions=predictions, method_unavailable=unavailable,
            fitted_maps=maps, hf_projection=projection, half_states=half_states,
            half_predictions=halves, elapsed_prediction=elapsed, elapsed_fit=timing,
            reliability=reliability, training_diagnostics={
                "native_units": len(plan["indices"]), "mean_count_per_bin": float(counts.values.mean()),
                "population_rank": context.numerical_rank,
                "pc2_variance_fraction": float(np.sum(context.singular_values[:2]**2)/np.sum(context.singular_values**2)),
                "target_change_is_not_artifact_cause": True})
    except Unavailable as error:
        cell["unavailable"] = str(error)
    return cell


def run(views_path, attempt, check):
    initial._compute_only()
    if attempt.parent.resolve() != ANALYSIS or not attempt.name.startswith("attempt"):
        raise PermissionError("fresh attempt must be under the declared EP06 analysis directory")
    ANALYSIS.mkdir(exist_ok=True)
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "declaration.json", {"program": program(), "views": str(views_path),
        "started_utc": initial._utc(), "job_id": os.environ.get("SLURM_JOB_ID")})
    with initial._new_file(attempt / "events.jsonl") as events:
        try:
            views = CandidateDevelopmentViews(views_path, check=check)
            metadata = {s.filename: views.metadata(s.filename, "M1", "training") for s in SESSIONS}
            plans = support_plan(metadata)
            initial._json_new(attempt / "support_plan.json", plans)
            original = original_cells(views_path, check)
            count = 0
            with initial._new_file(attempt / "predictions.pickle", binary=True) as stream:
                for spec in SESSIONS:
                    check()
                    train = views.training(spec.filename, "M1")
                    features = views.features(spec.filename, "M1", "evaluation")
                    em = views.metadata(spec.filename, "M1", "evaluation")
                    for plan in plans["plans"][spec.filename]:
                        cell = fit_cell(spec, plan, original[spec.filename], train, features, metadata[spec.filename], em, check)
                        initial._dump_cell(stream, cell)
                        count += 1
                    initial._event(events, "session_predictions_saved", session=spec.filename, completed_cells=count)
            initial._json_new(attempt / "fit_complete.json", {"cells": count, "utc": initial._utc()})
            # Only the durable all-session publication above admits this reader.
            outcomes = DevelopmentOutcomeViews(views_path, role="evaluation", check=check)
            rows = []
            with initial._owned_read(attempt / "predictions.pickle", binary=True) as stream:
                for si, spec in enumerate(SESSIONS):
                    counts = outcomes.targets(spec.filename, "M1")
                    mappings = mismatch_indices(counts.design, [6006302, si])
                    for plan in plans["plans"][spec.filename]:
                        cell = pickle.load(stream)
                        if cell["session"] != spec.filename or cell["plan"] != plan:
                            raise ValueError("published prediction order/support changed")
                        rows.append(score_cell(cell, counts, mappings, check))
                    initial._event(events, "session_scored", session=spec.filename)
                if stream.read(1):
                    raise ValueError("unexpected prediction cell")
            report = {"program": program(), "support": plans, "cells": rows, "summary": summary(rows),
                "completed_utc": initial._utc(), "primary_search_result": "no_supported_candidate_under_original_design",
                "claim_scope": "post_selection_exploratory_same_seven_dates_two_animals",
                "limitations": ["different_target_units_change_population_axes_and_energy",
                    "disjoint_electrodes_are_not_verified_spatially_distant",
                    "high_frequency_projection_can_remove_neural_signal_and_cannot_prove_contamination_absent",
                    "mapping_stability_is_not_measurement_reliability_or_a_noise_ceiling",
                    "mismatch_reference_is_conditional_not_selection_adjusted_inference"]}
            initial._json_new(attempt / "report.json", report)
            initial._event(events, "complete", cells=len(rows))
            return report
        except BaseException as error:
            initial._event(events, "failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1500)
    parser.add_argument("--rss-gib", type=float, default=5)
    args = parser.parse_args()
    initial._compute_only()
    if not 0 < args.max_seconds <= 1500 or not 0 < args.rss_gib <= 5:
        parser.error("bounded 1500-second/5-GiB process caps required")
    deadline = time.monotonic()+args.max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed follow-up deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.rss_gib*1024**2:
            raise MemoryError("fixed follow-up memory cap reached")
    run(args.views, args.attempt, check)


if __name__ == "__main__":
    main()
