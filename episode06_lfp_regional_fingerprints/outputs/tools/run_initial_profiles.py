"""First twelve EP06 development comparisons, with a durable outcome boundary.

``fit_predict`` reads training outcomes and evaluation features only. Six fit
families, each with fourteen session/region cells, supply all twelve recipes.
Only a fully published fit stage can enter ``score`` and construct the separate
evaluation-outcome reader. Calibration values and the four reserved sessions
are never requested. The HDF reader separation remains procedural within the
same account, not operating-system isolation.

The six pickle streams are trusted, locally produced state under one fixed
attempt path; they are not an interchange format. Exclusive creation preserves
every complete cell and an interrupted final record. There is no automatic
retry, overwrite, rank fallback, session deletion or fit after scoring. Reports
are development diagnostics: they neither complete required controls nor
nominate a final procedure, claim independent animals or stop the search.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np

import bandwise_prediction as prediction
from component_scoring import BoundActivity
from development_views import CandidateDevelopmentViews, DevelopmentOutcomeViews
from prepare_development_views import SESSIONS, REGIONS, _compute_only
from primary_profile import paired_profile, score_band_profile
from profile_classifier import ClassifierRecipe, PairedSessionProfiles
from profile_references import ASSIGNMENTS, evaluate_development_reference, signed_profile_agreement


EPISODE = Path(__file__).resolve().parents[2]
ATTEMPT_PATH = EPISODE / "outputs/runs/residual_fingerprint_development/starting001"
BLOCKS = tuple(prediction.FeatureBlock(name, (band,)) for name, band in zip(
    ("lmp", "power_0p5_4", "power_4_8", "power_8_12", "power_12_25",
     "power_25_50", "power_50_100", "power_100_200", "power_200_400"), prediction.BANDS))
FAMILIES = (
    ("starting_00", prediction.BandwiseRecipe(2, .01, BLOCKS)),
    ("starting_02", prediction.BandwiseRecipe(2, .001, BLOCKS)),
    ("starting_03", prediction.BandwiseRecipe(2, .1, BLOCKS)),
    ("starting_04", prediction.BandwiseRecipe(4, .01, BLOCKS)),
    ("starting_05", prediction.BandwiseRecipe(2, .01, BLOCKS, "median_ridge")),
    ("starting_09", prediction.BandwiseRecipe(2, .01, BLOCKS, "pca2_cca", .01)),
)


@dataclass(frozen=True)
class StartingCandidate:
    candidate_id: str
    family_id: str
    block_indices: tuple
    classifier: ClassifierRecipe


FULL, LOW = tuple(range(9)), tuple(range(6))
PROTOTYPE = ClassifierRecipe("cosine_prototype", None)
LOGISTIC1 = ClassifierRecipe("l2_logistic", 1.)
LOGISTIC01 = ClassifierRecipe("l2_logistic", .1)
CANDIDATES = tuple(StartingCandidate(f"starting_{index:02d}", family, indices, classifier)
    for index, (family, indices, classifier) in enumerate((
        ("starting_00", FULL, PROTOTYPE), ("starting_00", LOW, PROTOTYPE),
        ("starting_02", FULL, PROTOTYPE), ("starting_03", FULL, PROTOTYPE),
        ("starting_04", FULL, PROTOTYPE), ("starting_05", FULL, PROTOTYPE),
        ("starting_00", FULL, LOGISTIC1), ("starting_00", LOW, LOGISTIC1),
        ("starting_00", FULL, LOGISTIC01), ("starting_09", FULL, PROTOTYPE),
        ("starting_09", LOW, PROTOTYPE), ("starting_09", FULL, LOGISTIC1))))
ROSTERS = tuple(tuple(spec.filename for spec in SESSIONS if spec.animal == animal)
                for animal in ("Mihili", "Chewie-L"))


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned deadline/resource check required")
    check()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    """Use JSON's actual tuple/list representation in the frozen declaration."""
    return json.loads(json.dumps(value, allow_nan=False))


def _program():
    return _plain({"kind": "EP06_initial_twelve_development_profiles",
        "sessions": [{"animal": spec.animal, "session": spec.filename} for spec in SESSIONS],
        "regions": REGIONS, "fit_families": {name: asdict(recipe) for name, recipe in FAMILIES},
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "outcome_role": "evaluation", "calibration_values_used": False,
        "reserved_sessions_opened": False, "fit_cell_count": 84,
        "claim_scope": "development_diagnostics_only_required_controls_outstanding"})


def _attempt_path(path):
    path = Path(path)
    if not path.is_absolute() or path != ATTEMPT_PATH or path.resolve() != ATTEMPT_PATH:
        raise PermissionError("only the explicitly fixed starting001 attempt path is permitted")
    return path


def _new_file(path, *, binary=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"}))


def _json_new(path, value):
    with _new_file(path) as handle:
        json.dump(value, handle, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _event(handle, event, **fields):
    handle.write(json.dumps({"utc": _utc(), "event": event, **fields}, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _dump_cell(handle, cell):
    pickle.dump(cell, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.flush()
    os.fsync(handle.fileno())


def _validate_cell(cell, family, recipe, spec, region, *, check):
    """Bind every frozen cell before evaluation outcome construction is possible."""
    _check(check)
    if not isinstance(cell, dict) or set(cell) != {"family", "session", "region", "fit", "predictions"}:
        raise ValueError("one complete fit/prediction cell is required")
    if (cell["family"], cell["session"], cell["region"]) != (family, spec.filename, region):
        raise ValueError("fit stream does not match the exact ordered development roster")
    fitted, predicted = cell["fit"], cell["predictions"]
    if not isinstance(fitted, prediction.FrozenPopulationContext):
        raise TypeError("a frozen bandwise population fit is required")
    if (fitted.recipe != recipe or fitted.region != region
            or fitted.training_design.session_id != spec.filename
            or fitted.training_design.role != "training"
            or len(fitted.electrode_ids) != 15 or fitted.bands != prediction.BANDS):
        raise ValueError("fitted recipe, native feature support or training identity changed")
    if not isinstance(predicted, prediction.BandwisePredictions):
        raise TypeError("frozen explicitly bound predictions are required")
    if (predicted.region != region or predicted.design.session_id != spec.filename
            or predicted.design.role != "evaluation" or predicted.blocks != BLOCKS
            or not isinstance(predicted.predictions, tuple) or len(predicted.predictions) != 9):
        raise ValueError("all nine evaluation-only predictions must be frozen in canonical order")
    native_design = replace(predicted.design, axis_ids=fitted.training_design.axis_ids)
    if prediction._bind_fitted(fitted, native_design, region) != predicted.design:
        raise ValueError("prediction axes differ from the frozen population fit")
    for block in predicted.predictions:
        _check(check)
        if not isinstance(block, BoundActivity) or block.design != predicted.design:
            raise ValueError("every predicted block must have the complete frozen trial design")
        values = np.asarray(block.values)
        if values.shape != predicted.design.shape or not np.isfinite(values).all():
            raise ValueError("a prediction block is incomplete or nonfinite")


def fit_predict(*, views_path, attempt_dir=ATTEMPT_PATH, check):
    """Exclusive new attempt; no outcome-reader construction in this stage."""
    _compute_only()
    _check(check)
    attempt = _attempt_path(attempt_dir)
    views_path = Path(views_path)
    if not views_path.is_absolute() or views_path.name != "development_views.h5":
        raise ValueError("an explicit completed development_views.h5 path is required")
    attempt.mkdir(mode=0o700)  # Parent is prepared by the coordinating batch.
    _json_new(attempt / "recipes.json", {"program": _program(), "views_path": str(views_path)})
    with _new_file(attempt / "fit_status.jsonl") as status:
        _event(status, "fit_predict_started", job_id=os.environ.get("SLURM_JOB_ID"))
        complete = 0
        try:
            views = CandidateDevelopmentViews(views_path, check=check)
            for family, recipe in FAMILIES:
                with _new_file(attempt / f"{family}.pickle", binary=True) as stream:
                    for spec in SESSIONS:
                        for region in REGIONS:
                            _check(check)
                            lfp, training = views.training(spec.filename, region)
                            fitted = prediction.fit_bandwise(lfp, training, recipe=recipe, check=check)
                            features = views.features(spec.filename, region, "evaluation")
                            predicted = prediction.predict_bandwise(fitted, features, check=check)
                            cell = {"family": family, "session": spec.filename, "region": region,
                                    "fit": fitted, "predictions": predicted}
                            _validate_cell(cell, family, recipe, spec, region, check=check)
                            _dump_cell(stream, cell)
                            complete += 1
                            _event(status, "prediction_cell_published", family=family,
                                   session=spec.filename, region=region, completed_cells=complete)
            _check(check)
            if complete != 84:
                raise RuntimeError("the frozen twelve-comparison program requires all 84 fit cells")
            # Last publication is the sole transition to evaluation outcome access.
            _json_new(attempt / "fit_complete.json", {"completed_cells": 84,
                "families": [family for family, _ in FAMILIES], "completed_utc": _utc()})
            _event(status, "fit_predict_complete", completed_cells=complete)
        except BaseException as error:
            _event(status, "fit_predict_failed", completed_cells=complete,
                   error_type=type(error).__name__, error=str(error))
            raise
    return attempt


def _owned_read(path, *, binary=False):
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise PermissionError("only trusted nonsymlink local artifacts owned by this account may be read")
    return path.open("rb" if binary else "r", **({} if binary else {"encoding": "utf-8"}))


def _load_completed(attempt, *, check):
    """Check the complete roster and every payload before returning any model."""
    _check(check)
    with _owned_read(attempt / "fit_complete.json") as handle:
        completed = json.load(handle)
    if (completed.get("completed_cells") != 84
            or completed.get("families") != [family for family, _ in FAMILIES]):
        raise PermissionError("all six fitting families must be published before scoring")
    with _owned_read(attempt / "recipes.json") as handle:
        manifest = json.load(handle)
    if set(manifest) != {"program", "views_path"} or manifest["program"] != _program():
        raise ValueError("the frozen initial-twelve program differs from the published fit declaration")
    views_path = Path(manifest["views_path"])
    if not views_path.is_absolute() or views_path.name != "development_views.h5":
        raise ValueError("fit declaration has no permitted development view")
    cells = {}
    for family, recipe in FAMILIES:
        with _owned_read(attempt / f"{family}.pickle", binary=True) as stream:
            for spec in SESSIONS:
                for region in REGIONS:
                    _check(check)
                    cell = pickle.load(stream)  # Only our fixed-path, same-account producer.
                    _validate_cell(cell, family, recipe, spec, region, check=check)
                    cells[family, spec.filename, region] = cell
            if stream.read(1):
                raise ValueError("a fit family contains trailing or duplicate cells")
    return views_path, cells


def candidate_panels(base_profiles, candidate, *, check):
    """Select frozen profile blocks only; no neural refit or outcome access."""
    _check(check)
    if candidate not in CANDIDATES:
        raise ValueError("only the twelve prospectively fixed candidates are admitted")
    panels = ([], [])
    for spec in SESSIONS:
        _check(check)
        full = base_profiles[candidate.family_id, spec.filename]
        if full.session_id != spec.filename or full.blocks != BLOCKS:
            raise ValueError("base profile does not have its exact full-band session identity")
        m1 = replace(full.m1, blocks=tuple(full.m1.blocks[index] for index in candidate.block_indices))
        pmd = replace(full.pmd, blocks=tuple(full.pmd.blocks[index] for index in candidate.block_indices))
        profile = paired_profile(m1, pmd, check=check)
        panels[0 if spec.animal == "Mihili" else 1].append(PairedSessionProfiles(spec.animal, profile))
    return tuple(tuple(panel) for panel in panels)


def _profile_summary(supplied):
    profile = supplied.profile
    return {"animal": supplied.animal, "session": profile.session_id,
        "m1_scores": [value.score for value in profile.m1.blocks],
        "pmd_scores": [value.score for value in profile.pmd.blocks],
        "m1_residual_energy": profile.m1.residual_energy,
        "pmd_residual_energy": profile.pmd.residual_energy,
        "m1_minus_pmd": profile.m1_minus_pmd, "available": profile.available,
        "unavailable_reasons": profile.unavailable_reasons}


def evaluate_candidate(base_profiles, candidate, *, check):
    panels = candidate_panels(base_profiles, candidate, check=check)
    reference = evaluate_development_reference(*panels, mihili_sessions=ROSTERS[0],
        chewie_sessions=ROSTERS[1], recipe=candidate.classifier, check=check)
    signed = signed_profile_agreement(*panels, mihili_sessions=ROSTERS[0],
        chewie_sessions=ROSTERS[1], check=check)
    observed = tuple(direction.observed_balanced_accuracy for direction in reference.directions)
    mean_accuracy = math.fsum(observed) / 2. if all(value is not None for value in observed) else None
    return {"candidate": asdict(candidate), "directional_balanced_accuracy": observed,
        "mean_balanced_accuracy": mean_accuracy,
        "mean_balanced_accuracy_minus_half": None if mean_accuracy is None else mean_accuracy - .5,
        "profiles": [_profile_summary(value) for panel in panels for value in panel],
        "joint_label_reference": asdict(reference), "signed_agreement": asdict(signed),
        "claim_scope": "development_diagnostics_only_required_controls_outstanding",
        "scientific_acceptance": None, "search_stop_decision": None}


def fixed_menu_reference(reports, *, check):
    """Replay only the fixed twelve-candidate maximum-mean choice, without fits.

Every candidate participates at every assignment; an unavailable candidate is
never excluded to manufacture a complete menu. Exact ties retain the lowest
starting ID. This diagnostic adjusts this fixed max-mean menu only, not later
adaptive choices, separate directional tests, controls or biological claims.
"""
    _check(check)
    if tuple(report["candidate"]["candidate_id"] for report in reports) != tuple(
            candidate.candidate_id for candidate in CANDIDATES):
        raise ValueError("the complete ordered first-twelve menu is required")
    references = tuple(report["joint_label_reference"] for report in reports)
    for reference in references:
        if (tuple(tuple(bits) for bits in reference["assignments"]) != ASSIGNMENTS
                or len(reference["directions"]) != 2
                or any(len(direction["assignment_balanced_accuracies"]) != 128
                       for direction in reference["directions"])):
            raise ValueError("menu candidates must share all 128 exact joint label assignments")
        if tuple((direction["source_animal"], direction["target_animal"])
                 for direction in reference["directions"]) != (
                     ("Mihili", "Chewie-L"), ("Chewie-L", "Mihili")):
            raise ValueError("menu reference directions must preserve the physical source/target identities")
    selections, unavailable = [], []
    for index in range(128):
        _check(check)
        best, missing = None, []
        for candidate, reference in zip(CANDIDATES, references):
            accuracies = tuple(direction["assignment_balanced_accuracies"][index]
                               for direction in reference["directions"])
            if any(value is None or not math.isfinite(value) or not 0. <= value <= 1.
                   for value in accuracies):
                missing.append(candidate.candidate_id)
                continue
            mean = math.fsum(accuracies) / 2.
            if best is None or mean > best["mean_balanced_accuracy"]:
                best = {"candidate_id": candidate.candidate_id,
                    "directional_balanced_accuracy": accuracies,
                    "mean_balanced_accuracy": mean, "mean_balanced_accuracy_minus_half": mean - .5}
        selections.append(None if missing else best)
        unavailable.append(tuple(missing))
    complete = not any(unavailable) and all(reference["complete"] for reference in references)
    count = (sum(value["mean_balanced_accuracy"] >= selections[0]["mean_balanced_accuracy"]
                 for value in selections) if complete else None)
    return {"complete": complete, "assignments": ASSIGNMENTS, "selections": selections,
        "observed_selection": selections[0], "unavailable_candidates_by_assignment": unavailable,
        "incomplete_candidate_references": [candidate.candidate_id for candidate, reference
            in zip(CANDIDATES, references) if not reference["complete"]],
        "inclusive_upper_tail_count": count,
        "inclusive_upper_tail_fraction": count / 128. if complete else None,
        "unavailable_reason": None if complete else "complete_first_twelve_menu_reference_unavailable",
        "claim_scope": "fixed_first_twelve_maximum_mean_choice_only_no_adaptive_or_population_inference"}


def score(*, attempt_dir=ATTEMPT_PATH, check):
    """One score invocation after all first-twelve fit/predictions are fixed."""
    _compute_only()
    _check(check)
    attempt = _attempt_path(attempt_dir)
    views_path, cells = _load_completed(attempt, check=check)
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)  # No overwrite or implicit score-stage retry.
    with _new_file(destination / "status.jsonl") as status:
        _event(status, "score_started", job_id=os.environ.get("SLURM_JOB_ID"))
        count = 0
        try:
            # This constructor occurs only after ALL 84 payloads have validated.
            outcomes = DevelopmentOutcomeViews(views_path, role="evaluation", check=check)
            regional, base_profiles = {}, {}
            with _new_file(destination / "regional_scores.pickle", binary=True) as stream:
                for spec in SESSIONS:
                    for region in REGIONS:
                        _check(check)
                        observed = outcomes.targets(spec.filename, region)
                        for family, _ in FAMILIES:
                            cell = cells[family, spec.filename, region]
                            result = score_band_profile(cell["fit"], observed, cell["predictions"], check=check)
                            regional[family, spec.filename, region] = result
                            _dump_cell(stream, {"family": family, "session": spec.filename,
                                                "region": region, "profile": result})
                            count += 1
                        _event(status, "regional_scores_published", session=spec.filename,
                               region=region, completed_cells=count)
            for family, _ in FAMILIES:
                for spec in SESSIONS:
                    base_profiles[family, spec.filename] = paired_profile(
                        regional[family, spec.filename, "M1"], regional[family, spec.filename, "PMd"], check=check)
            reports, menu_reports = [], []
            for candidate in CANDIDATES:
                _check(check)
                report = evaluate_candidate(base_profiles, candidate, check=check)
                _json_new(destination / f"{candidate.candidate_id}.json", report)
                menu_reports.append(report)
                reports.append({"candidate_id": candidate.candidate_id,
                    "directional_balanced_accuracy": report["directional_balanced_accuracy"],
                    "mean_balanced_accuracy_minus_half": report["mean_balanced_accuracy_minus_half"],
                    "reference_complete": report["joint_label_reference"]["complete"],
                    "raw_signed_available": report["signed_agreement"]["raw_available"],
                    "raw_signed_all_positive": report["signed_agreement"]["raw_all_positive"]})
                _event(status, "candidate_report_published", candidate_id=candidate.candidate_id)
            _check(check)
            menu = fixed_menu_reference(menu_reports, check=check)
            _json_new(destination / "fixed_menu_reference.json", menu)
            _json_new(destination / "complete.json", {"completed_utc": _utc(),
                "fit_cells": 84, "scored_cells": count, "candidates": reports,
                "fixed_menu_reference_complete": menu["complete"],
                "claim_scope": "development_diagnostics_only_required_controls_outstanding",
                "scientific_acceptance": None, "search_stop_decision": None})
            _event(status, "score_complete", completed_candidates=len(reports))
        except BaseException as error:
            _event(status, "score_failed", scored_cells=count,
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
            parser.error("score reads the exact fit-stage view declaration; do not override --views")
        score(attempt_dir=args.attempt_dir, check=check)


if __name__ == "__main__":
    main()
