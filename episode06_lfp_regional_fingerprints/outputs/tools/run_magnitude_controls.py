"""Six fixed EP06 score-reuse comparisons; no neural fitting or outcome reads.

The original/adaptive score streams and completed control JSONs are trusted
local artifacts. Control JSONs omit designs, so only metadata from their own
completed fit/prediction streams supplies those designs. In particular the
selected-unit population is never replaced by the original population. Saved
physical scores and residual energies remain unchanged; omitted squared errors
are explicitly unavailable rather than reconstructed measurements.

Four controls and two successors receive complete 128-assignment references
and raw signed-profile diagnostics. These exploratory comparisons neither
reset patience nor open calibration/reserved data or authorize promotion.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import time

import bandwise_prediction as prediction
from component_scoring import FrozenDesign
from primary_profile import BlockScore, ProfileScore, paired_profile
from profile_classifier import PairedSessionProfiles, PairedLabels, score_profile_predictions
from profile_references import ASSIGNMENTS, evaluate_development_reference, signed_profile_agreement
from magnitude_classifier import (evaluate_magnitude_reference, fit_magnitude_classifier,
                                  predict_magnitude_classifier)
import run_initial_profiles as initial
import run_followup_profiles as followup
import run_development_controls as controls
import run_support_controls as support


ATTEMPT = initial.ATTEMPT_PATH.parent / "representation001"
SOURCE_DIRS = {"original": initial.ATTEMPT_PATH, "adaptive": followup.ATTEMPT_PATH,
               "mismatch": controls.ATTEMPT, "selected_units": support.ATTEMPT}


@dataclass(frozen=True)
class Comparison:
    candidate_id: str
    category: str
    source: str
    family: str
    block_indices: tuple
    representation: str
    comparator: str


COMPARISONS = (
    Comparison("control06normalization", "control", "original", "starting_09", initial.FULL,
               "shape_plus_signed_mean", "starting_11"),
    Comparison("control07lmp", "control", "original", "starting_09", (0,),
               "signed_mean_only", "starting_10"),
    Comparison("control08selectedunits", "control", "selected_units", "starting_09", (0,),
               "signed_mean_only", "control07lmp"),
    Comparison("control09mismatch", "control", "mismatch", "starting_09", (0,),
               "signed_mean_only", "control07lmp"),
    Comparison("adaptive02", "successor", "adaptive", "adaptive00", (0,),
               "signed_mean_only", "starting_10"),
    Comparison("adaptive03", "successor", "original", "starting_09", (0, 2, 3, 4, 5),
               "centered_unit_l2_shape_prototype", "starting_10"),
)


def _program():
    return initial._plain({"kind": "EP06_six_fixed_score_reuse_comparisons",
        "comparisons": [asdict(value) for value in COMPARISONS],
        "sources": {name: str(path) for name, path in SOURCE_DIRS.items()},
        "sessions": [{"animal": spec.animal, "session": spec.filename} for spec in initial.SESSIONS],
        "regions": initial.REGIONS, "magnitude_penalty": 1.,
        "magnitude_location": "source_row_signed_mean_median",
        "magnitude_scale": "1.4826_times_source_row_signed_mean_MAD_no_fallback",
        "new_neural_fits": 0, "new_outcome_reads": 0, "controls": 4, "successors": 2,
        "calibration_values_used": False, "reserved_sessions_opened": False,
        "local_stage_seconds_max": 600, "global_clock_reset": False})


def _read_json(path):
    with initial._owned_read(path) as handle:
        return json.load(handle)


def _completed_sources(check):
    """Admit all fixed completed stages before reading any score payload."""
    manifests = {name: _read_json(path / "recipes.json") for name, path in SOURCE_DIRS.items()}
    programs = {"original": initial._program(), "adaptive": followup._program(),
                "mismatch": controls._program(), "selected_units": support._program()}
    view = manifests["original"]["views_path"]
    for name, manifest in manifests.items():
        check()
        if manifest != {"program": programs[name], "views_path": view}:
            raise ValueError("all reused score producers must retain their exact recipe and development view")
    done = {name: _read_json(path / "scores/complete.json") for name, path in SOURCE_DIRS.items()}
    for name, count, ids in (("original", 84, [value.candidate_id for value in initial.CANDIDATES]),
                             ("adaptive", 28, [value[0] for value in followup.FAMILIES])):
        if (done[name].get("fit_cells") != count or done[name].get("scored_cells") != count
                or [row["candidate_id"] for row in done[name].get("candidates", ())] != ids):
            raise PermissionError("complete exact initial and successor score stages are required")
    for name in ("mismatch", "selected_units"):
        if (done[name].get("calibration_values_used") is not False
                or done[name].get("reserved_sessions_opened") is not False):
            raise PermissionError("score reuse cannot broaden the outcome roles")
    if any(not value.get("completed_utc") for value in done.values()):
        raise PermissionError("all four original score stages must be completed")
    return done, view


def _profile_identity(profile, spec, region, blocks):
    if (not isinstance(profile, ProfileScore) or profile.region != region
            or not isinstance(profile.design, FrozenDesign)
            or profile.design.session_id != spec.filename or profile.design.role != "evaluation"
            or len(profile.design.axis_ids) != 2
            or tuple(value.block for value in profile.blocks) != blocks):
        raise ValueError("scored profiles must retain exact development session/region/rank/blocks")


def _score_stream(source, families, wanted, check):
    result = {}
    with initial._owned_read(SOURCE_DIRS[source] / "scores/regional_scores.pickle", binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                for family in families:
                    check()
                    cell = pickle.load(stream)
                    if (set(cell) != {"family", "session", "region", "profile"}
                            or (cell["family"], cell["session"], cell["region"]) != (family, spec.filename, region)):
                        raise ValueError("complete ordered regional score stream required")
                    if family == wanted:
                        _profile_identity(cell["profile"], spec, region, initial.BLOCKS)
                        result[spec.filename, region] = cell["profile"]
        if stream.read(1):
            raise ValueError("unexpected extra score cells")
    return result


def _control_designs(source, check):
    """Read actual control designs from saved state without inspecting arrays."""
    path = SOURCE_DIRS[source]
    done = _read_json(path / "fit_complete.json")
    if source == "mismatch":
        if done.get("cells") != 14:
            raise PermissionError("mismatch prediction stage is incomplete")
        filename = "mismatch.pickle"
    else:
        if done.get("unit_control_available") is not True or done.get("unit_cells") != 14:
            return None
        filename = "selected_units.pickle"
    designs = {}
    with initial._owned_read(path / filename, binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                check()
                cell = pickle.load(stream)
                if (set(cell) != {"family", "session", "region", "fit", "predictions"}
                        or (cell["family"], cell["session"], cell["region"]) != ("starting_09", spec.filename, region)):
                    raise ValueError("control state must retain exact ordered identities")
                fitted, predicted = cell["fit"], cell["predictions"]
                if (not isinstance(fitted, prediction.FittedCCABandwise)
                        or not isinstance(predicted, prediction.BandwisePredictions)
                        or fitted.recipe != dict(initial.FAMILIES)["starting_09"]
                        or fitted.region != region or predicted.region != region
                        or predicted.blocks != initial.BLOCKS
                        or fitted.training_design.session_id != spec.filename
                        or fitted.training_design.role != "training"
                        or predicted.design.session_id != spec.filename or predicted.design.role != "evaluation"
                        or predicted.design.axis_ids != fitted.population_axis_ids
                        or len(predicted.design.axis_ids) != 2):
                    raise ValueError("control design does not belong to its actual frozen population context")
                native = replace(predicted.design, axis_ids=fitted.training_design.axis_ids)
                if prediction._bind_fitted(fitted, native, region) != predicted.design:
                    raise ValueError("control evaluation design is not bound to its own fitted state")
                designs[spec.filename, region] = predicted.design
        if stream.read(1):
            raise ValueError("unexpected additional control state")
    return designs


def _control_profiles(source, report, check):
    if report.get("status") == "unavailable":
        return None
    if report.get("candidate") != initial._plain(asdict(initial.CANDIDATES[10])):
        raise ValueError("control score summaries require the unchanged original low-six recipe")
    if tuple(report["joint_label_reference"]["blocks"]) != tuple(initial._plain(
            [asdict(block) for block in initial.BLOCKS[:6]])):
        raise ValueError("control score summaries do not retain the physical low-six block roster")
    designs = _control_designs(source, check)
    if designs is None:
        return None
    rows = report["profiles"]
    if len(rows) != 7:
        raise ValueError("all seven control session summaries are required")
    result = {}
    for spec, row in zip(initial.SESSIONS, rows):
        if (row["animal"], row["session"]) != (spec.animal, spec.filename):
            raise ValueError("control summaries changed session order or animal identity")
        for region in initial.REGIONS:
            values, energy = row[f"{region.lower()}_scores"], row[f"{region.lower()}_residual_energy"]
            if len(values) != 6:
                raise ValueError("each control summary needs all six physical block positions")
            if energy is not None and (not isinstance(energy, Real) or isinstance(energy, bool)
                                       or not math.isfinite(energy) or energy < 0):
                raise ValueError("stored residual energy must be finite nonnegative or unavailable")
            if (energy is None or energy == 0) and any(value is not None for value in values):
                raise ValueError("a finite stored physical score requires positive residual energy")
            blocks = tuple(BlockScore(block, None, value, None if value is not None
                                      else "stored_regional_score_unavailable")
                           for block, value in zip(initial.BLOCKS[:6], values))
            profile = ProfileScore(region, designs[spec.filename, region], energy, blocks)
            _profile_identity(profile, spec, region, initial.BLOCKS[:6])
            result[spec.filename, region] = profile
    return result


def _panels(regional, indices, check):
    if regional is None:
        return tuple(tuple(None for _ in roster) for roster in initial.ROSTERS)
    result = ([], [])
    for spec in initial.SESSIONS:
        pair = [replace(regional[spec.filename, region], blocks=tuple(
            regional[spec.filename, region].blocks[index] for index in indices)) for region in initial.REGIONS]
        profile = paired_profile(*pair, check=check)
        result[int(spec.animal != "Mihili")].append(PairedSessionProfiles(spec.animal, profile))
    return tuple(tuple(panel) for panel in result)


def _score_grids(report, blocks):
    """The existing paired-reference rule, allowing this candidate's own blocks."""
    reference = report["joint_label_reference"]
    directions = reference["directions"]
    if (tuple(tuple(bits) for bits in reference["assignments"]) != ASSIGNMENTS
            or initial._plain(reference["blocks"]) != initial._plain([asdict(block) for block in blocks])
            or len(directions) != 2):
        raise ValueError("exact complete assignment slots and declared physical blocks required")
    for index, direction in enumerate(directions):
        animals = ("Mihili", "Chewie-L")
        if (direction["source_animal"] != animals[index] or direction["target_animal"] != animals[1-index]
                or tuple(direction["source_sessions"]) != initial.ROSTERS[index]
                or tuple(direction["target_sessions"]) != initial.ROSTERS[1-index]
                or len(direction["assignment_balanced_accuracies"]) != 128):
            raise ValueError("paired references must retain both complete development rosters")
    def grid(values):
        counts = []
        for value, denominator in zip(values, (16, 12)):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("finite complete balanced accuracy or explicit unavailability required")
            integer = round(value * denominator)
            if abs(value - integer / denominator) > 8 * math.ulp(1.):
                raise ValueError("accuracy changed the half-credit session grid")
            counts.append(integer)
        return 3 * counts[0] + 4 * counts[1] - 48
    observed = grid([row["observed_balanced_accuracy"] for row in directions])
    scores = tuple(grid([row["assignment_balanced_accuracies"][index] for row in directions]) for index in range(128))
    if observed != scores[0]:
        raise ValueError("ordinary label assignment differs from observed accuracy")
    return observed, scores, reference["complete"] is True and all(value is not None for value in scores)


def _compare_successor(report, baseline, comparison, check):
    check()
    observed, scores, complete = _score_grids(report, tuple(initial.BLOCKS[i] for i in comparison.block_indices))
    before, original, base_complete = _score_grids(baseline, initial.BLOCKS[:6])
    delta = None if observed is None or before is None else observed-before
    paired = tuple(None if a is None or b is None else a-b for a, b in zip(scores, original))
    complete = complete and base_complete
    median_twice = noise_twice = None
    if complete:
        ordered = sorted(paired)
        median_twice = ordered[63] + ordered[64]
        noise_twice = sorted(abs(2*value-median_twice) for value in paired)[121]
    one_step = None if delta is None else delta >= 1
    above_noise = None if not complete else 2*delta > noise_twice
    return {"candidate_id": comparison.candidate_id, "incumbent_id": "starting_10",
        "complete_paired_reference": complete, "score_grid_denominator": 96,
        "observed_grid_improvement": delta, "observed_improvement": None if delta is None else delta/96.,
        "paired_reference_differences": [None if value is None else value/96. for value in paired],
        "median_paired_difference": None if median_twice is None else median_twice/192.,
        "noise_tolerance": None if noise_twice is None else noise_twice/192.,
        "at_least_one_grid_step": one_step, "strictly_above_noise": above_noise,
        "numerical_improvement_pass": None if not complete else bool(one_step and above_noise),
        "applicable_controls_passed": None, "material_improvement": None,
        "patience_reset_authorized": False, "claim_scope": "search_heuristic_only_not_significance_or_stopping"}


def _evaluate(comparison, regional, baseline, check):
    panels = _panels(regional, comparison.block_indices, check)
    kwargs = {"mihili_sessions": initial.ROSTERS[0], "chewie_sessions": initial.ROSTERS[1], "check": check}
    if comparison.representation == "centered_unit_l2_shape_prototype":
        recipe = initial.PROTOTYPE
        reference = evaluate_development_reference(*panels, recipe=recipe, **kwargs)
    else:
        # The magnitude classifier infers shape+mean for K>1 and mean alone
        # for K=1 from these fixed physical blocks; both use lambda=1.
        recipe = initial.LOGISTIC1
        reference = evaluate_magnitude_reference(*panels, recipe=recipe, **kwargs)
    signed = signed_profile_agreement(*panels, **kwargs)
    values = tuple(row.observed_balanced_accuracy for row in reference.directions)
    mean = math.fsum(values)/2. if all(value is not None for value in values) else None
    report = {"candidate": {**asdict(comparison), "classifier": asdict(recipe)},
        "directional_balanced_accuracy": values, "mean_balanced_accuracy": mean,
        "mean_balanced_accuracy_minus_half": None if mean is None else mean-.5,
        "profiles": [initial._profile_summary(row) for panel in panels for row in panel if row is not None],
        "profile_designs": [{"session": row.profile.session_id,
            "M1": asdict(row.profile.m1.design), "PMd": asdict(row.profile.pmd.design)}
            for panel in panels for row in panel if row is not None],
        "joint_label_reference": asdict(reference), "signed_agreement": asdict(signed),
        "source_profile_unavailable": regional is None,
        "source_attempt": str(SOURCE_DIRS[comparison.source]),
        "squared_error_provenance": "stored_original_pickle_or_unavailable_in_control_JSON_no_reconstruction",
        "claim_scope": "exploratory_fixed_score_reuse_no_selection_adjusted_significance",
        "scientific_acceptance": None, "search_stop_decision": None,
        "patience_reset_authorized": False}
    if comparison.category == "successor":
        report["comparison_to_starting10"] = _compare_successor(report, baseline, comparison, check)
    if comparison.representation != "centered_unit_l2_shape_prototype":
        # Retain the already-successful ordinary-label source models for
        # interpretation. This deterministic replay adds no candidate/null
        # assignment and never retries an unavailable observed fit.
        fitted_reports = []
        for index, direction in enumerate(reference.directions):
            if direction.observed_balanced_accuracy is None:
                fitted_reports.append({"source_animal": direction.source_animal, "status": "unavailable",
                    "reason": direction.unavailable_reason})
                continue
            labels = tuple(PairedLabels(direction.source_animal, session, 1, 0)
                           for session in initial.ROSTERS[index])
            fitted = fit_magnitude_classifier(panels[index], source_animal=direction.source_animal,
                expected_sessions=initial.ROSTERS[index], labels=labels, recipe=recipe, check=check)
            predicted = predict_magnitude_classifier(fitted, panels[1-index],
                target_animal=direction.target_animal, expected_sessions=initial.ROSTERS[1-index], check=check)
            observed_score = score_profile_predictions(predicted, labels=tuple(
                PairedLabels(direction.target_animal, session, 1, 0)
                for session in initial.ROSTERS[1-index]), check=check)
            if observed_score.balanced_accuracy != direction.observed_balanced_accuracy:
                raise ValueError("retained ordinary-label model changed its reference prediction")
            fitted_report = asdict(fitted)
            fitted_report["coefficients"] = fitted.coefficients.tolist()
            fitted_reports.append({"status": "complete", "fit": fitted_report,
                "target_predictions": asdict(predicted), "target_score": asdict(observed_score)})
        report["observed_magnitude_classifiers"] = fitted_reports
        report["observed_model_replay_fit_count"] = sum(row["status"] == "complete" for row in fitted_reports)
        report["observed_model_replay_additional_scientific_trials"] = 0
    return report


def _physical_comparisons(original, mismatch, elapsed, check):
    if elapsed.get("control_id") != "control05_elapsed_time" or len(elapsed.get("cells", ())) != 14:
        raise ValueError("exact fourteen-cell elapsed-time report required")
    rows = []
    def same_energy(left, right):
        if left is None or right is None:
            return left is right
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=0)
    for index, (spec, region) in enumerate((s, r) for s in initial.SESSIONS for r in initial.REGIONS):
        check()
        full, timing = original[spec.filename, region], elapsed["cells"][index]
        if (timing["animal"], timing["session"], timing["region"]) != (spec.animal, spec.filename, region):
            raise ValueError("time-baseline report changed its paired session support")
        if not same_energy(full.residual_energy, timing["residual_energy"]):
            raise ValueError("elapsed and original scores require identical population residual energy")
        other = None if mismatch is None else mismatch[spec.filename, region]
        if other is not None and (other.design != full.design
                                  or not same_energy(other.residual_energy, full.residual_energy)):
            raise ValueError("trial mismatch must preserve the original target design and population energy")
        observed = full.blocks[0].score
        shifted = None if other is None else other.blocks[0].score
        timed = timing["score"]
        rows.append({"animal": spec.animal, "session": spec.filename, "region": region,
            "block": asdict(initial.BLOCKS[0]), "original_lmp": observed,
            "mismatched_lmp": shifted, "elapsed_time": timed,
            "original_lmp_minus_mismatched_lmp": None if observed is None or shifted is None else observed-shifted,
            "original_lmp_minus_elapsed_time": None if observed is None or timed is None else observed-timed,
            "residual_energy": full.residual_energy})
    return {"cells": rows, "additional_scientific_trial_count": 0,
        "claim_scope": "named_physical_score_differences_only_no_conditional_information_or_inferential_claim"}


def run(*, attempt_dir=ATTEMPT, deadline_epoch):
    initial._compute_only()
    if not math.isfinite(deadline_epoch):
        raise ValueError("finite coordinator-owned absolute deadline required")
    local_deadline = min(deadline_epoch, time.time()+600.)
    def check():
        if time.time() >= local_deadline:
            raise TimeoutError("shared resource deadline or 600-second local stage cap reached")
    check()
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or attempt != ATTEMPT or attempt.resolve() != ATTEMPT:
        raise PermissionError("only the fixed representation001 attempt is permitted")
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "recipes.json", {"program": _program(),
        "coordinator_deadline_epoch": deadline_epoch, "effective_deadline_epoch": local_deadline})
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)
    with initial._new_file(destination / "status.jsonl") as status:
        initial._event(status, "score_reuse_started", job_id=os.environ.get("SLURM_JOB_ID"))
        summaries = []
        try:
            _completed_sources(check)
            original = _score_stream("original", [name for name, _ in initial.FAMILIES], "starting_09", check)
            adaptive = _score_stream("adaptive", [name for name, _ in followup.FAMILIES], "adaptive00", check)
            source = {"original": original, "adaptive": adaptive}
            for name, identifier in (("mismatch", "control03_mismatch"),
                                      ("selected_units", "control04_selected_electrode_units")):
                report = _read_json(SOURCE_DIRS[name] / f"scores/{identifier}.json")
                if report.get("control_id") != identifier:
                    raise ValueError("completed control report identity changed")
                source[name] = _control_profiles(name, report, check)
            baseline = _read_json(initial.ATTEMPT_PATH / "scores/starting_10.json")
            normalized = _read_json(initial.ATTEMPT_PATH / "scores/starting_11.json")
            for report, index in ((baseline, 10), (normalized, 11)):
                if report["candidate"] != initial._plain(asdict(initial.CANDIDATES[index])):
                    raise ValueError("original nominal comparator identity changed")
            reports = {}
            for comparison in COMPARISONS:
                check()
                report = _evaluate(comparison, source[comparison.source], baseline, check)
                comparator = (normalized if comparison.comparator == "starting_11" else baseline
                              if comparison.comparator == "starting_10" else reports[comparison.comparator])
                before, after = comparator["directional_balanced_accuracy"], report["directional_balanced_accuracy"]
                report["nominal_comparator"] = {"candidate_id": comparison.comparator,
                    "directional_balanced_accuracy": before,
                    "directional_accuracy_change": [None if a is None or b is None else a-b for a, b in zip(after, before)],
                    "claim_scope": "descriptive_comparison_not_control_pass"}
                if comparison.candidate_id == "adaptive02":
                    direct = reports["control07lmp"]["directional_balanced_accuracy"]
                    report["same_representation_mapping_comparator"] = {
                        "candidate_id": "control07lmp", "directional_balanced_accuracy": direct,
                        "directional_accuracy_change": [None if a is None or b is None else a-b
                                                        for a, b in zip(after, direct)],
                        "claim_scope": "descriptive_PCA2_ridge_vs_CCA_with_identical_singleton_LMP_logistic_representation"}
                initial._json_new(destination / f"{comparison.candidate_id}.json", report)
                reports[comparison.candidate_id] = report
                summaries.append({"candidate_id": comparison.candidate_id, "category": comparison.category,
                    "directional_balanced_accuracy": after,
                    "reference_complete": report["joint_label_reference"]["complete"],
                    "raw_signed_available": report["signed_agreement"]["raw_available"],
                    "raw_signed_all_positive": report["signed_agreement"]["raw_all_positive"]})
                initial._event(status, "comparison_published", candidate_id=comparison.candidate_id)
            timing = _read_json(support.ATTEMPT / "scores/control05_elapsed_time.json")
            physical = _physical_comparisons(original, source["mismatch"], timing, check)
            initial._json_new(destination / "physical_score_comparisons.json", physical)
            check()
            initial._json_new(destination / "complete.json", {"completed_utc": initial._utc(),
                "comparisons": summaries, "controls_attempted": 4, "successors_attempted": 2,
                "valid_scientific_comparison_count": sum(row["reference_complete"] and row["raw_signed_available"]
                                                         for row in summaries),
                "valid_control_count": sum(row["category"] == "control" and row["reference_complete"]
                                           and row["raw_signed_available"] for row in summaries),
                "valid_successor_count": sum(row["category"] == "successor" and row["reference_complete"]
                                             and row["raw_signed_available"] for row in summaries),
                "new_neural_fits": 0, "new_outcome_reads": 0,
                "calibration_values_used": False, "reserved_sessions_opened": False,
                "required_controls_passed": None, "scientific_acceptance": None,
                "search_stop_decision": None, "patience_reset_authorized": False})
            initial._event(status, "score_reuse_complete", completed_comparisons=6)
        except BaseException as error:
            initial._event(status, "score_reuse_failed", completed_comparisons=len(summaries),
                           error_type=type(error).__name__, error=str(error))
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, default=ATTEMPT)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    run(attempt_dir=args.attempt_dir, deadline_epoch=args.deadline_epoch)


if __name__ == "__main__":
    main()
