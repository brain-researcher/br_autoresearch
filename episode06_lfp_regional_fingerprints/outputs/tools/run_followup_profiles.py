"""Two fixed EP06 successor comparisons, without empirical control execution.

adaptive00 uses electrode-PCA2/ridge(.01); adaptive01 uses electrode-PCA2/CCA
with covariance regularization .1 and prediction ridge .01. Both use spike
rank2, full-nine frozen singleton predictions, low-six scored profiles and the
cosine prototype classifier. Both complete fourteen fit/prediction cells before
the score stage can read evaluation targets. Calibration/reserved values are
never requested. As in the initial runner, separation is procedural within the
same account, and pickle streams are trusted local state under one fixed path.

The paired material-improvement calculation reuses the initial starting_10
report. Its numerical threshold is a search heuristic, not a significance
test. Controls remain outstanding: this runner never resets patience, opens
reserved sessions, grants scientific acceptance or chooses a next candidate.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import time

import run_initial_profiles as initial
import bandwise_prediction as prediction
from development_views import CandidateDevelopmentViews, DevelopmentOutcomeViews
from primary_profile import paired_profile, score_band_profile
from profile_classifier import PairedSessionProfiles
from profile_references import ASSIGNMENTS, evaluate_development_reference, signed_profile_agreement
from prepare_development_views import SESSIONS, REGIONS, _compute_only


ATTEMPT_PATH = initial.ATTEMPT_PATH.parent / "adaptive001"
BASELINE_PATH = initial.ATTEMPT_PATH / "scores/starting_10.json"
FAMILIES = (
    ("adaptive00", prediction.BandwiseRecipe(2, .01, initial.BLOCKS, "pca2_ridge")),
    ("adaptive01", prediction.BandwiseRecipe(2, .01, initial.BLOCKS, "pca2_cca", .1)),
)


def _declaration(candidate_id, recipe):
    return {"candidate_id": candidate_id, "fit_recipe": asdict(recipe),
            "block_indices": initial.LOW, "classifier": asdict(initial.PROTOTYPE)}


def _program():
    return initial._plain({"kind": "EP06_two_development_successors",
        "sessions": [{"animal": spec.animal, "session": spec.filename} for spec in SESSIONS],
        "regions": REGIONS, "candidates": [_declaration(*family) for family in FAMILIES],
        "baseline_report": str(BASELINE_PATH), "fit_cell_count": 28,
        "outcome_role": "evaluation", "calibration_values_used": False,
        "reserved_sessions_opened": False, "empirical_controls_executed": False,
        "claim_scope": "development_diagnostics_only_required_controls_outstanding"})


def _attempt_path(path):
    path = Path(path)
    if not path.is_absolute() or path != ATTEMPT_PATH or path.resolve() != ATTEMPT_PATH:
        raise PermissionError("only the fixed adaptive001 attempt path is permitted")
    return path


def fit_predict(*, views_path, attempt_dir=ATTEMPT_PATH, check):
    _compute_only()
    initial._check(check)
    attempt = _attempt_path(attempt_dir)
    views_path = Path(views_path)
    if not views_path.is_absolute() or views_path.name != "development_views.h5":
        raise ValueError("an explicit completed development_views.h5 path is required")
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "recipes.json", {"program": _program(), "views_path": str(views_path)})
    with initial._new_file(attempt / "fit_status.jsonl") as status:
        initial._event(status, "fit_predict_started", job_id=os.environ.get("SLURM_JOB_ID"))
        complete = 0
        try:
            views = CandidateDevelopmentViews(views_path, check=check)
            for candidate_id, recipe in FAMILIES:
                with initial._new_file(attempt / f"{candidate_id}.pickle", binary=True) as stream:
                    for spec in SESSIONS:
                        for region in REGIONS:
                            initial._check(check)
                            lfp, training = views.training(spec.filename, region)
                            fitted = prediction.fit_bandwise(lfp, training, recipe=recipe, check=check)
                            features = views.features(spec.filename, region, "evaluation")
                            predicted = prediction.predict_bandwise(fitted, features, check=check)
                            cell = {"family": candidate_id, "session": spec.filename,
                                    "region": region, "fit": fitted, "predictions": predicted}
                            initial._validate_cell(cell, candidate_id, recipe, spec, region, check=check)
                            initial._dump_cell(stream, cell)
                            complete += 1
                            initial._event(status, "prediction_cell_published", candidate_id=candidate_id,
                                session=spec.filename, region=region, completed_cells=complete)
            initial._check(check)
            if complete != 28:
                raise RuntimeError("both prospective successors require all 28 fit/prediction cells")
            initial._json_new(attempt / "fit_complete.json", {"completed_cells": 28,
                "families": [name for name, _ in FAMILIES], "completed_utc": initial._utc()})
            initial._event(status, "fit_predict_complete", completed_cells=complete)
        except BaseException as error:
            initial._event(status, "fit_predict_failed", completed_cells=complete,
                           error_type=type(error).__name__, error=str(error))
            raise
    return attempt


def _load_completed(attempt, *, check):
    initial._check(check)
    with initial._owned_read(attempt / "fit_complete.json") as handle:
        completed = json.load(handle)
    if (completed.get("completed_cells") != 28
            or completed.get("families") != [name for name, _ in FAMILIES]):
        raise PermissionError("both complete successor prediction families are required before scoring")
    with initial._owned_read(attempt / "recipes.json") as handle:
        manifest = json.load(handle)
    if set(manifest) != {"program", "views_path"} or manifest["program"] != _program():
        raise ValueError("the successor recipes differ from the published fit declaration")
    views_path = Path(manifest["views_path"])
    if not views_path.is_absolute() or views_path.name != "development_views.h5":
        raise ValueError("the fit declaration has no permitted development view")
    cells = {}
    for candidate_id, recipe in FAMILIES:
        with initial._owned_read(attempt / f"{candidate_id}.pickle", binary=True) as stream:
            for spec in SESSIONS:
                for region in REGIONS:
                    initial._check(check)
                    cell = pickle.load(stream)
                    initial._validate_cell(cell, candidate_id, recipe, spec, region, check=check)
                    cells[candidate_id, spec.filename, region] = cell
            if stream.read(1):
                raise ValueError("a successor family contains trailing or duplicate cells")
    return views_path, cells


def _reference_grids(report):
    reference = report["joint_label_reference"]
    if (tuple(tuple(bits) for bits in reference["assignments"]) != ASSIGNMENTS
            or initial._plain(reference["blocks"]) != initial._plain([asdict(b) for b in initial.BLOCKS[:6]])
            or len(reference["directions"]) != 2):
        raise ValueError("the fixed low-six reference with all 128 assignments is required")
    directions = reference["directions"]
    animals = ("Mihili", "Chewie-L")
    for index, direction in enumerate(directions):
        if (direction["source_animal"] != animals[index]
                or direction["target_animal"] != animals[1 - index]
                or tuple(direction["source_sessions"]) != initial.ROSTERS[index]
                or tuple(direction["target_sessions"]) != initial.ROSTERS[1 - index]
                or len(direction["assignment_balanced_accuracies"]) != 128):
            raise ValueError("comparison directions must retain their exact three/four-session support")

    def units(value, denominator):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("balanced accuracy must be numeric or explicitly unavailable")
        if not math.isfinite(value):
            return None
        if not 0. <= value <= 1.:
            raise ValueError("balanced accuracy lies outside its possible support")
        integer = round(float(value) * denominator)
        # Serialization rounding only: decisions below use exact integer
        # counts, never a tolerance around the scientific improvement boundary.
        if abs(float(value) - integer / denominator) > 8 * math.ulp(1.):
            raise ValueError("balanced accuracy does not lie on the complete half-credit session grid")
        return integer

    def score_grid(values):
        # Four target pairs give BA units1/16, three give1/12. Their
        # equal-mean-minus-half score is exactly (3*k4+4*k3-48)/96.
        a, b = (units(value, denominator) for value, denominator in zip(values, (16, 12)))
        return None if a is None or b is None else 3 * a + 4 * b - 48

    observed = score_grid(tuple(direction["observed_balanced_accuracy"] for direction in directions))
    scores = tuple(score_grid(tuple(direction["assignment_balanced_accuracies"][index]
                                    for direction in directions)) for index in range(128))
    if scores[0] is not None and observed is not None and scores[0] != observed:
        raise ValueError("the ordinary-label assignment differs from the observed score")
    complete = reference["complete"] is True and observed is not None and all(value is not None for value in scores)
    return observed, scores, complete


def compare_to_starting10(report, baseline, *, check):
    """Exact-grid implementation of the adopted paired-reference heuristic."""
    initial._check(check)
    candidate_id = report["candidate"]["candidate_id"]
    if candidate_id not in tuple(name for name, _ in FAMILIES) or baseline["candidate"]["candidate_id"] != "starting_10":
        raise ValueError("one adopted successor and the fixed starting_10 incumbent are required")
    observed, scores, complete = _reference_grids(report)
    base_observed, base_scores, base_complete = _reference_grids(baseline)
    delta = None if observed is None or base_observed is None else observed - base_observed
    paired = tuple(None if a is None or b is None else a - b for a, b in zip(scores, base_scores))
    complete = complete and base_complete
    median_twice = noise_twice = None
    if complete:
        ordered = sorted(paired)
        median_twice = ordered[63] + ordered[64]
        noise_twice = sorted(abs(2 * value - median_twice) for value in paired)[121]
    one_step = None if delta is None else delta >= 1
    above_noise = None if not complete else 2 * delta > noise_twice
    initial._check(check)
    return {"candidate_id": candidate_id, "incumbent_id": "starting_10",
        "complete_paired_reference": complete, "score_grid_denominator": 96,
        "observed_grid_improvement": delta,
        "observed_improvement": None if delta is None else delta / 96.,
        "paired_reference_differences": tuple(None if value is None else value / 96. for value in paired),
        "median_paired_difference": None if median_twice is None else median_twice / 192.,
        "noise_tolerance": None if noise_twice is None else noise_twice / 192.,
        "at_least_one_grid_step": one_step, "strictly_above_noise": above_noise,
        "numerical_improvement_pass": None if not complete else bool(one_step and above_noise),
        "applicable_controls_passed": None, "material_improvement": None,
        "patience_reset_authorized": False,
        "unavailable_reason": None if complete else "complete_paired_candidate_incumbent_reference_unavailable",
        "claim_scope": "search_heuristic_only_controls_outstanding_not_significance_or_stopping"}


def _load_baseline(views_path, *, check):
    initial._check(check)
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as handle:
        manifest = json.load(handle)
    if manifest.get("program") != initial._program() or manifest.get("views_path") != str(views_path):
        raise ValueError("successors and the incumbent must use the same frozen development view")
    with initial._owned_read(initial.ATTEMPT_PATH / "scores/complete.json") as handle:
        completed = json.load(handle)
    if (completed.get("fit_cells") != 84 or completed.get("scored_cells") != 84
            or tuple(row["candidate_id"] for row in completed.get("candidates", ())) != tuple(
                candidate.candidate_id for candidate in initial.CANDIDATES)):
        raise PermissionError("the complete initial-twelve score stage is required")
    with initial._owned_read(BASELINE_PATH) as handle:
        baseline = json.load(handle)
    if baseline.get("candidate") != initial._plain(asdict(initial.CANDIDATES[10])):
        raise ValueError("the incumbent must be the unchanged starting_10 low-six CCA prototype")
    _reference_grids(baseline)
    return baseline


def evaluate_candidate(regional, candidate_id, recipe, baseline, *, check):
    panels = ([], [])
    for spec in SESSIONS:
        pair = []
        for region in REGIONS:
            full = regional[candidate_id, spec.filename, region]
            if tuple(block.block for block in full.blocks) != initial.BLOCKS:
                raise ValueError("the full nine frozen scores are required before low-six projection")
            pair.append(replace(full, blocks=full.blocks[:6]))
        profile = paired_profile(*pair, check=check)
        if profile.session_id != spec.filename:
            raise ValueError("profile session differs from the complete development roster")
        panels[0 if spec.animal == "Mihili" else 1].append(PairedSessionProfiles(spec.animal, profile))
    panels = tuple(tuple(panel) for panel in panels)
    reference = evaluate_development_reference(*panels, mihili_sessions=initial.ROSTERS[0],
        chewie_sessions=initial.ROSTERS[1], recipe=initial.PROTOTYPE, check=check)
    signed = signed_profile_agreement(*panels, mihili_sessions=initial.ROSTERS[0],
        chewie_sessions=initial.ROSTERS[1], check=check)
    observed = tuple(direction.observed_balanced_accuracy for direction in reference.directions)
    mean = math.fsum(observed) / 2. if all(value is not None for value in observed) else None
    report = {"candidate": _declaration(candidate_id, recipe),
        "directional_balanced_accuracy": observed, "mean_balanced_accuracy": mean,
        "mean_balanced_accuracy_minus_half": None if mean is None else mean - .5,
        "profiles": [initial._profile_summary(value) for panel in panels for value in panel],
        "joint_label_reference": asdict(reference), "signed_agreement": asdict(signed),
        "claim_scope": "development_diagnostics_only_required_controls_outstanding",
        "scientific_acceptance": None, "search_stop_decision": None}
    report["comparison_to_starting10"] = compare_to_starting10(report, baseline, check=check)
    return report


def score(*, attempt_dir=ATTEMPT_PATH, check):
    _compute_only()
    initial._check(check)
    attempt = _attempt_path(attempt_dir)
    views_path, cells = _load_completed(attempt, check=check)
    baseline = _load_baseline(views_path, check=check)
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)
    with initial._new_file(destination / "status.jsonl") as status:
        initial._event(status, "score_started", job_id=os.environ.get("SLURM_JOB_ID"))
        count = 0
        try:
            # No outcome reader is constructed until all28 new cells and the
            # fixed incumbent declaration have validated.
            outcomes = DevelopmentOutcomeViews(views_path, role="evaluation", check=check)
            regional = {}
            with initial._new_file(destination / "regional_scores.pickle", binary=True) as stream:
                for spec in SESSIONS:
                    for region in REGIONS:
                        initial._check(check)
                        observed = outcomes.targets(spec.filename, region)
                        for candidate_id, _ in FAMILIES:
                            cell = cells[candidate_id, spec.filename, region]
                            result = score_band_profile(cell["fit"], observed, cell["predictions"], check=check)
                            regional[candidate_id, spec.filename, region] = result
                            initial._dump_cell(stream, {"family": candidate_id, "session": spec.filename,
                                                       "region": region, "profile": result})
                            count += 1
                        initial._event(status, "regional_scores_published", session=spec.filename,
                                       region=region, completed_cells=count)
            summaries = []
            for candidate_id, recipe in FAMILIES:
                report = evaluate_candidate(regional, candidate_id, recipe, baseline, check=check)
                initial._json_new(destination / f"{candidate_id}.json", report)
                summaries.append({"candidate_id": candidate_id,
                    "directional_balanced_accuracy": report["directional_balanced_accuracy"],
                    "mean_balanced_accuracy_minus_half": report["mean_balanced_accuracy_minus_half"],
                    "reference_complete": report["joint_label_reference"]["complete"],
                    "raw_signed_available": report["signed_agreement"]["raw_available"],
                    "raw_signed_all_positive": report["signed_agreement"]["raw_all_positive"],
                    "comparison_to_starting10": report["comparison_to_starting10"]})
                initial._event(status, "candidate_report_published", candidate_id=candidate_id)
            initial._check(check)
            initial._json_new(destination / "complete.json", {"completed_utc": initial._utc(),
                "fit_cells": 28, "scored_cells": count, "candidates": summaries,
                "empirical_controls_executed": False,
                "scientific_acceptance": None, "search_stop_decision": None})
            initial._event(status, "score_complete", completed_candidates=2)
        except BaseException as error:
            initial._event(status, "score_failed", scored_cells=count,
                           error_type=type(error).__name__, error=str(error))
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit_predict", "score"), required=True)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--attempt-dir", type=Path, default=ATTEMPT_PATH)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    _compute_only()
    if not math.isfinite(args.deadline_epoch):
        parser.error("--deadline-epoch must be finite")
    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("coordinator-owned absolute resource deadline reached")
    if args.mode == "fit_predict":
        if args.views is None:
            parser.error("fit_predict requires --views")
        fit_predict(views_path=args.views, attempt_dir=args.attempt_dir, check=check)
    else:
        if args.views is not None:
            parser.error("score uses the exact fit-stage view declaration; do not override --views")
        score(attempt_dir=args.attempt_dir, check=check)


if __name__ == "__main__":
    main()
