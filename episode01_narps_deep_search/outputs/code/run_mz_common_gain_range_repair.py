#!/usr/bin/env python3
"""One fresh common-gain component after exact first-M0500 exhaustion.

M0 changes only its numerical sweep ceiling to1500; the previously unattempted
MZ retains500. Each selected component uses the original full source/replay,
actual filtered rows and score sums/counts through the unchanged common runner.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qualify_assets import require_compute
import run_mz_common_gain_range as common
import summarize_mz_common_gain_range as original_pool


SOURCE, IMPLEMENTATION = common.SOURCE, common.IMPLEMENTATION
FAILED_JOB = "46613139"
FAILED_ROOT = SOURCE.parent / f"20261005-MZ-common-gain-range-{FAILED_JOB}"
PHASE = "MZ0_common_gain_single_model_after_M0_500_exhaustion"
REPORT_NAME = "mz_common_gain_range_repair.json"
MAX_SECONDS = 4200
IDENTITIES = {
    0: {"finished_utc": "2026-10-05T12:22:35.145654+00:00",
        "elapsed_seconds": 960.493597004097, "fit_elapsed_seconds": 959.1470382090192,
        "tail": [5862.903850910726, 5862.9032289695015]},
    1: {"finished_utc": "2026-10-05T12:22:48.871898+00:00",
        "elapsed_seconds": 974.2173690970521, "fit_elapsed_seconds": 972.8723926700186,
        "tail": [5733.553283596222, 5733.552109814011]},
}
NONCONVERGENCE = "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"
_ORIGINAL_PHASE, _ORIGINAL_MODELS, _ORIGINAL_CAP = common.PHASE, common.MODELS, common.MAX_SECONDS
_validate_original_fit = common.validate_fit


def failed_path(direction):
    common.selected_runs(direction)
    return FAILED_ROOT / f"dir{direction}" / "mz_common_gain_range.json"


def runtime_specification(original, model):
    if model not in ("M0", "MZ") or original["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Only the original500 M0/MZ components are selected")
    runtime = deepcopy(original)
    if model == "M0":
        runtime["optimizer"]["maximum_sweeps"] = 1500
    return runtime


def numerical_override(model):
    if model not in ("M0", "MZ"):
        raise ValueError("Only M0 or previously unattempted MZ is selected")
    return {"parameter": "optimizer.maximum_sweeps", "original_value": 500,
        "runtime_value": 1500 if model == "M0" else 500,
        "changed": model == "M0", "scope": "one_fresh_model_one_direction",
        "likelihood_priors_support_starts_tolerances_and_first_convergence_changed": False,
        "nonconverged_optimizer_state_reused": False}


def repair_context(failed, source, people, spec, direction):
    """Validate this exact failed attempt; people supplies source identity keys.

    The producer additionally recomputes its selected rows from actual saved
    events and compares retained support before its new fit. A pool can use the
    independently validated source score keys here without reloading events.
    """
    train, held = common.selected_runs(direction)
    identity = IDENTITIES[direction]
    if (failed.get("finished_utc") != identity["finished_utc"]
            or failed.get("elapsed_seconds") != identity["elapsed_seconds"]
            or failed.get("status") != "technical_common_gain_failure"
            or failed.get("error") != NONCONVERGENCE
            or failed.get("internal_seconds") != 5400
            or failed.get("source_baseline_validated") is not True
            or failed.get("saved_primary_scores_replayed") is not True
            or len(failed.get("fits", [])) != 1 or failed.get("scored_fits") != []
            or failed.get("completed_model_scores") != {}):
        raise ValueError("Require the exact terminal first-M0 failure, not an alternate or partial attempt")
    # The original report remains its actual two-model500 phase. Restore only
    # validator bindings while this process's single-model adapter is active.
    with ExitStack() as stack:
        for name, value in (("PHASE", _ORIGINAL_PHASE), ("MODELS", _ORIGINAL_MODELS),
                            ("MAX_SECONDS", _ORIGINAL_CAP), ("validate_fit", _validate_original_fit)):
            stack.enter_context(patch.object(common, name, value))
        _, panel, primary, source_identity, support = original_pool.validate_component(failed, direction, FAILED_JOB, spec)
    if (panel is not None or source_identity != (str(source["job_id"]), source["finished_utc"])
            or primary != source["participant_scores"] or set(primary) != set(people)
            or set(support) != set(people)):
        raise ValueError("The failed direction must retain this exact full source, primary last bits and participants")
    fit = failed["fits"][0]
    names = ["kappa", "log_a", "nondecision_logit"]
    if (fit.get("parameter_names") != names or fit.get("version_order") != ["EI", "ER"]
            or fit.get("inference") != spec["method"] or fit.get("converged") is not False
            or fit.get("subject_optimizer_failures") != []
            or fit.get("elapsed_seconds") != identity["fit_elapsed_seconds"]):
        raise ValueError("Only the diagnosed M0 numerical sweep exhaustion is eligible")
    parameters = np.asarray(fit.get("parameters"), dtype=float)
    if parameters.shape != (108, 3) or not np.isfinite(parameters).all():
        raise ValueError("The preserved failed parameters must remain finite and complete")
    for column, name in enumerate(names):
        low, high = spec["parameters"][name]["bounds"]
        if np.any(parameters[:, column] < low) or np.any(parameters[:, column] > high):
            raise ValueError("The failed parameters left the original support")
    for field, shape in (("global_mean", (3,)), ("version_means", (2, 3)), ("version_standard_deviations", (2, 3))):
        values = np.asarray(fit.get(field), dtype=float)
        if (values.shape != shape or not np.isfinite(values).all()
                or field == "version_standard_deviations" and np.any(values <= 0)):
            raise ValueError("Incomplete failed hyperparameter evidence")
    history = np.asarray(fit.get("objective_history"), dtype=float)
    if history.shape != (500,) or not np.isfinite(history).all() or history[-2:].tolist() != identity["tail"]:
        raise ValueError("Retain the exact finite500-sweep exhaustion history")
    gap = float(abs(history[-1] - history[-2]))
    threshold = spec["optimizer"]["joint_relative_tolerance"] * (1 + abs(float(history[-2])))
    if not gap > threshold:
        raise ValueError("The old fit must fail the unchanged first-convergence criterion")
    contacts = fit.get("support_contacts")
    if (not isinstance(contacts, list) or len(contacts) != 2
            or {row.get("version") for row in contacts} != {"EI", "ER"}
            or any(row.get("level") != "version_log_standard_deviation" or row.get("parameter") != "kappa"
                or row.get("bounds") != [-4., 1.] or row.get("value") != -3.9999998503354774 for row in contacts)
            or spec["parameters"]["kappa"]["individual_log_scale_bounds"] != [-4., 1.]):
        raise ValueError("Preserve both exact lower kappa-scale contacts; numerical repair does not remove them")
    return {"kind": "first_M0_original500_sweep_exhaustion_split_into_fresh_single_model_components",
        "direction": direction, "failed_report": str(failed_path(direction)), "failed_job_id": FAILED_JOB,
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": failed["finished_utc"],
        "failed_error": failed["error"], "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": fit["elapsed_seconds"], "previous_support_contacts": deepcopy(contacts),
        "retained_support": deepcopy(support), "final_objective_change": gap,
        "unchanged_convergence_threshold": threshold, "original_maximum_sweeps": 500,
        "nonconverged_checkpoint_state_reused": False,
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "cost_accounting": "This direction's prior failed process is one shared lineage reference for M0 and MZ, counted once when pooling; it is not a new charge in either component. Parent allocation accounting is separate."}


def run(args):
    require_compute()
    common.selected_runs(args.direction)
    if args.model not in ("M0", "MZ"):
        raise ValueError("Select only M0 or previously unattempted MZ")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The entire single-model process is capped at4200 seconds")
    output = args.output_dir.resolve()
    if output == FAILED_ROOT.resolve() or FAILED_ROOT.resolve() in output.parents:
        raise FileExistsError("Preserve the original failed direction outputs")
    base_validate, base_fit, base_writer = common.shared.validate_source, common.fit_partially_pooled, common.write_report
    base_filter = common.filtered_panels
    state = {"validated": False, "context": None, "calls": 0}
    model = args.model

    def source_validation(source, truth, people, spec):
        base_validate(source, truth, people, spec)
        failed = json.loads(failed_path(args.direction).read_text())
        state["context"] = repair_context(failed, source, people, spec, args.direction)
        state["validated"] = True

    def fitter(training, selected, rival, spec):
        if not state["validated"] or selected != model or rival != "fixed" or state["calls"]:
            raise ValueError("Exactly one fresh selected-model fit may follow validated source/failure context")
        state["calls"] += 1
        return base_fit(training, selected, rival, runtime_specification(spec, model))

    def filtered(people, direction):
        result = base_filter(people, direction)
        if not state["validated"] or result[2] != state["context"]["retained_support"]:
            raise ValueError("Newly filtered actual source rows must match the exact failed direction's retained slots/counts")
        return result

    def validator(fit, selected, train, held, spec, people):
        if selected != model:
            raise ValueError("The selected component cannot fit a different model")
        return _validate_original_fit(fit, selected, train, held, runtime_specification(spec, model), people)

    def writer(directory, report):
        spec = report.get("implementation")
        report.update(phase=PHASE, fitted_model=model, failed_report=str(failed_path(args.direction)),
            execution_kind="M0_numerical_repair" if model == "M0" else "previously_unattempted_MZ_component",
            source_context_validated=state["validated"], prior_failed_context=state["context"],
            original_implementation=spec,
            runtime_implementation=runtime_specification(spec, model) if spec is not None else None,
            original_maximum_sweeps=500, fitted_maximum_sweeps=1500 if model == "M0" else 500,
            numerical_override=numerical_override(model), fresh_fit_calls=state["calls"], retry_wave_selected=False,
            prior_direction_process_elapsed_seconds=state["context"]["previous_process_elapsed_seconds"] if state["validated"] else None,
            cost_accounting="elapsed_seconds is only this new single-model process. The prior failed direction is a shared reference, counted once across its M0/MZ pair; do not sum prior references or charge the original parent again.",
            interpretation="One fresh common-gain model/direction after preserved first-M0 exhaustion. M0 changes only its sweep ceiling500-to1500; previously unattempted MZ remains500. No saved fit or failed optimizer state is reused. Original full-source/replay, filtered trial slots, likelihood, priors, starts, tolerances and first convergence are unchanged. Held-out sums/counts remain unpooled; this supplies no recovery, family, robustness, interval or gate verdict.")
        base_writer(directory, report)

    with ExitStack() as stack:
        for name, value in (("MODELS", (model,)), ("PHASE", PHASE), ("REPORT_NAME", REPORT_NAME),
                            ("MAX_SECONDS", MAX_SECONDS), ("fit_partially_pooled", fitter),
                            ("filtered_panels", filtered),
                            ("validate_fit", validator), ("write_report", writer)):
            stack.enter_context(patch.object(common, name, value))
        stack.enter_context(patch.object(common.shared, "validate_source", source_validation))
        return common.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=int, choices=(0, 1), required=True)
    parser.add_argument("--model", choices=("M0", "MZ"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "direction", "fitted_model", "status",
        "common_gain_component_complete", "fresh_fit_calls", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["common_gain_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
