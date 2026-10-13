"""Four prospectively selected EP06 development controls on starting_10.

fit_predict freezes paired within-direction donor maps and all fourteen
mismatched prediction cells. score then reads evaluation outcomes, reuses the
completed original regional scores, and runs metadata/session/band controls.
No calibration values, reserved dates, source MATLAB or sibling scores enter.
This is a same-account procedural boundary, not OS isolation.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pickle
import time

import bandwise_prediction as prediction
from development_interventions import make_within_direction_mapping, apply_paired_lfp_mismatch
from development_views import CandidateDevelopmentViews, DevelopmentOutcomeViews
from metadata_classifier import evaluate_metadata_classifier, MetadataUnavailable
from primary_profile import paired_profile, score_band_profile
from profile_controls import evaluate_profile_controls
import run_initial_profiles as initial


ATTEMPT = initial.EPISODE / "outputs/runs/residual_fingerprint_development/controls001"
REPORT = initial.EPISODE / "outputs/runs/source_session_support/development001/report.json"
FAMILY = "starting_09"
RECIPE = dict(initial.FAMILIES)[FAMILY]
BASELINE = initial.CANDIDATES[10]


def _attempt(path):
    path = Path(path)
    if not path.is_absolute() or path != ATTEMPT or path.resolve() != ATTEMPT:
        raise PermissionError("only the declared controls001 attempt is permitted")
    return path


def _program():
    return initial._plain({"kind": "EP06_first_four_development_controls",
        "baseline": asdict(BASELINE), "recipe": asdict(RECIPE),
        "sessions": [asdict(spec) for spec in initial.SESSIONS],
        "controls": ["control00_metadata", "control01_leaveout", "control02_band", "control03_mismatch"],
        "mismatch_seed": [6006201, "animal_index", "original_position", "role_index"],
        "calibration_values_used": False, "reserved_sessions_opened": False})


def fit_predict(*, views_path, attempt_dir=ATTEMPT, check):
    initial._compute_only()
    check()
    attempt = _attempt(attempt_dir)
    views_path = Path(views_path)
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as handle:
        original = json.load(handle)
    if str(views_path) != original["views_path"] or original["program"] != initial._program():
        raise ValueError("controls require the same exact development export as initial comparisons")
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "recipes.json", {"program": _program(), "views_path": str(views_path)})
    views = CandidateDevelopmentViews(views_path, check=check)
    mappings = {}
    # Declare every mapping from metadata before any neural values are read.
    for spec in initial.SESSIONS:
        for role, role_index in (("training", 0), ("evaluation", 2)):
            left = views.metadata(spec.filename, "M1", role).design
            right = views.metadata(spec.filename, "PMd", role).design
            fields = ("session_id", "role", "directions", "trial_ids", "trial_directions", "time_ids")
            if any(getattr(left, name) != getattr(right, name) for name in fields):
                raise ValueError("mismatch requires exact paired trial/time metadata")
            mappings[spec.filename, role] = make_within_direction_mapping(
                session_id=left.session_id, role=role, directions=left.directions,
                trial_ids=left.trial_ids, trial_directions=left.trial_directions,
                seed=(6006201, int(spec.animal != "Mihili"), spec.original_position, role_index), check=check)
    initial._json_new(attempt / "donor_mappings.json", [asdict(value) for value in mappings.values()])
    complete = 0
    with initial._new_file(attempt / "fit_status.jsonl") as status:
        initial._event(status, "fit_predict_started", job_id=os.environ.get("SLURM_JOB_ID"))
        try:
            with initial._new_file(attempt / "mismatch.pickle", binary=True) as stream:
                for spec in initial.SESSIONS:
                    training = [views.training(spec.filename, region) for region in initial.REGIONS]
                    train_lfp = apply_paired_lfp_mismatch(training[0][0], training[1][0],
                        mapping=mappings[spec.filename, "training"], check=check)
                    features = [views.features(spec.filename, region, "evaluation") for region in initial.REGIONS]
                    test_lfp = apply_paired_lfp_mismatch(*features,
                        mapping=mappings[spec.filename, "evaluation"], check=check)
                    for index, region in enumerate(initial.REGIONS):
                        check()
                        fitted = prediction.fit_bandwise(train_lfp[index], training[index][1], recipe=RECIPE, check=check)
                        predicted = prediction.predict_bandwise(fitted, test_lfp[index], check=check)
                        cell = {"family": FAMILY, "session": spec.filename, "region": region,
                                "fit": fitted, "predictions": predicted}
                        initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                        initial._dump_cell(stream, cell)
                        complete += 1
                        initial._event(status, "prediction_published", session=spec.filename, region=region, cells=complete)
            if complete != 14:
                raise RuntimeError("all fourteen mismatch prediction cells required")
            initial._json_new(attempt / "fit_complete.json", {"cells": complete, "completed_utc": initial._utc()})
            initial._event(status, "fit_predict_complete", cells=complete)
        except BaseException as error:
            initial._event(status, "fit_predict_failed", cells=complete, error_type=type(error).__name__, error=str(error))
            raise


def _load_mismatch(attempt, check):
    with initial._owned_read(attempt / "fit_complete.json") as handle:
        if json.load(handle).get("cells") != 14:
            raise PermissionError("all fourteen predictions must precede outcome construction")
    with initial._owned_read(attempt / "recipes.json") as handle:
        declaration = json.load(handle)
    if declaration["program"] != _program():
        raise ValueError("control declaration changed")
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as handle:
        if declaration["views_path"] != json.load(handle)["views_path"]:
            raise ValueError("initial and control development views differ")
    cells = {}
    with initial._owned_read(attempt / "mismatch.pickle", binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                cell = pickle.load(stream)  # Fixed-path trusted local producer only.
                initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                cells[spec.filename, region] = cell
        if stream.read(1):
            raise ValueError("unexpected trailing mismatch cells")
    return Path(declaration["views_path"]), cells


def _original_panels(check):
    score_dir = initial.ATTEMPT_PATH / "scores"
    with initial._owned_read(score_dir / "complete.json") as handle:
        complete = json.load(handle)
    if complete["scored_cells"] != 84 or len(complete["candidates"]) != 12:
        raise PermissionError("original first twelve scores must be complete")
    regional = {}
    with initial._owned_read(score_dir / "regional_scores.pickle", binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                for family, _ in initial.FAMILIES:
                    check()
                    cell = pickle.load(stream)
                    if (cell["family"], cell["session"], cell["region"]) != (family, spec.filename, region):
                        raise ValueError("original regional-score identity/order differs")
                    if family == FAMILY:
                        regional[spec.filename, region] = cell["profile"]
        if stream.read(1):
            raise ValueError("unexpected original score cells")
    paired = {(FAMILY, spec.filename): paired_profile(regional[spec.filename, "M1"],
        regional[spec.filename, "PMd"], check=check) for spec in initial.SESSIONS}
    panels = initial.candidate_panels(paired, BASELINE, check=check)
    with initial._owned_read(score_dir / "starting_10.json") as handle:
        baseline = json.load(handle)
    if baseline["candidate"] != initial._plain(asdict(BASELINE)):
        raise ValueError("starting_10 baseline identity differs")
    return panels, baseline


def score(*, attempt_dir=ATTEMPT, check):
    initial._compute_only()
    check()
    attempt = _attempt(attempt_dir)
    views_path, cells = _load_mismatch(attempt, check)
    panels, baseline = _original_panels(check)
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)
    with initial._new_file(destination / "status.jsonl") as status:
        initial._event(status, "score_started", job_id=os.environ.get("SLURM_JOB_ID"))
        try:
            with initial._owned_read(REPORT) as handle:
                metadata = json.load(handle)
            try:
                metadata_report = evaluate_metadata_classifier(metadata, check=check)
            except MetadataUnavailable as error:
                metadata_report = {"status": "unavailable", "reason": str(error)}
            metadata_report["baseline_directional_balanced_accuracy"] = baseline["directional_balanced_accuracy"]
            if metadata_report["status"] == "complete":
                metadata_report["at_least_baseline_in_both_directions"] = all(
                    row["balanced_accuracy"] >= value for row, value in zip(
                        metadata_report["directions"], baseline["directional_balanced_accuracy"]))
            initial._json_new(destination / "control00_metadata.json", metadata_report)
            profile_reports = evaluate_profile_controls(*panels, mihili_sessions=initial.ROSTERS[0],
                chewie_sessions=initial.ROSTERS[1], recipe=BASELINE.classifier, check=check)
            profile_reports["baseline_directional_balanced_accuracy"] = baseline["directional_balanced_accuracy"]
            initial._json_new(destination / "profile_controls.json", profile_reports)
            # This constructor follows publication/validation of all14 predictions.
            outcomes = DevelopmentOutcomeViews(views_path, role="evaluation", check=check)
            paired = {}
            for spec in initial.SESSIONS:
                profiles = []
                for region in initial.REGIONS:
                    check()
                    cell = cells[spec.filename, region]
                    observed = outcomes.targets(spec.filename, region)
                    profiles.append(score_band_profile(cell["fit"], observed, cell["predictions"], check=check))
                paired[FAMILY, spec.filename] = paired_profile(*profiles, check=check)
            mismatch = initial.evaluate_candidate(paired, BASELINE, check=check)
            mismatch["control_id"] = "control03_mismatch"
            mismatch["baseline_directional_balanced_accuracy"] = baseline["directional_balanced_accuracy"]
            mismatch["directional_accuracy_change"] = [None if after is None else after - before
                for before, after in zip(baseline["directional_balanced_accuracy"], mismatch["directional_balanced_accuracy"])]
            mismatch["claim_scope"] = "single_fixed_pairing_sensitivity_not_significance_or_control_pass"
            initial._json_new(destination / "control03_mismatch.json", mismatch)
            available = {"control00_metadata": metadata_report["status"] == "complete",
                "control01_leaveout": profile_reports["leave_one_out"]["complete"],
                "control02_band": profile_reports["band_permutation"]["complete"],
                "control03_mismatch": mismatch["joint_label_reference"]["complete"]
                    and mismatch["signed_agreement"]["raw_available"]}
            initial._json_new(destination / "complete.json", {"completed_utc": initial._utc(),
                "four_controls_attempted": True, "available_controls": available,
                "valid_scientific_control_count": sum(available.values()),
                "metadata_available": metadata_report["status"] == "complete",
                "profile_controls_report": "profile_controls.json",
                "mismatch_reference_complete": mismatch["joint_label_reference"]["complete"],
                "mismatch_directional_accuracy": mismatch["directional_balanced_accuracy"],
                "required_controls_passed": None, "scientific_acceptance": None,
                "reserved_sessions_opened": False, "calibration_values_used": False})
            initial._event(status, "score_complete")
        except BaseException as error:
            initial._event(status, "score_failed", error_type=type(error).__name__, error=str(error))
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit_predict", "score"), required=True)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--attempt-dir", type=Path, default=ATTEMPT)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    initial._compute_only()
    if not math.isfinite(args.deadline_epoch):
        parser.error("finite absolute deadline required")
    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("coordinator-owned absolute resource deadline reached")
    if args.mode == "fit_predict":
        if args.views is None:
            parser.error("fit_predict requires --views")
        fit_predict(views_path=args.views, attempt_dir=args.attempt_dir, check=check)
    else:
        if args.views is not None:
            parser.error("score must use its published development view")
        score(attempt_dir=args.attempt_dir, check=check)


if __name__ == "__main__":
    main()
