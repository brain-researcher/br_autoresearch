#!/usr/bin/env python3
"""Repair only null43/61/168 first-half1000 MVC exhaustion, with fresh1500 fits.

The checked indexed loop owns its finite process timer, source/replay guards,
fitting, scoring and failure checkpoints. Its on-disk code remains unchanged.
Failed optimizer state is never reused; simulation and baseline fits stay unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_indexed_mvc as indexed
import summarize_mz_null_indexed_menu as admission


MAX_SECONDS, RUNTIME_SWEEPS = 12600, 1500
IDENTITIES = {
    43: {"failed_job_id": "46592703", "failed_finished_utc": "2026-10-05T08:39:21.255477+00:00",
         "source_job_id": "46580480", "source_finished_utc": "2026-10-05T06:25:28.258038+00:00"},
    61: {"failed_job_id": "46592721", "failed_finished_utc": "2026-10-05T08:45:14.649202+00:00",
         "source_job_id": "46580485", "source_finished_utc": "2026-10-05T06:00:20.322030+00:00"},
    168: {"failed_job_id": "46592827", "failed_finished_utc": "2026-10-05T08:42:30.877776+00:00",
          "source_job_id": "46581752", "source_finished_utc": "2026-10-05T06:48:38.827537+00:00"},
}
QUALIFYING_REPORT = admission.RUNS / "20261005-MZ42-MVC-reverse-completion-46587905/mz_mvc_reverse_completion.json"
REPORT_NAME = "mz_null_indexed_mvc_repair.json"
PHASE = "M0_null_indexed_MVC_runtime1500_first_half_repair"
NONCONVERGENCE = "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"


def failed_path(index):
    if type(index) is not int or index not in IDENTITIES:
        raise ValueError("Only the separately selected null43/61/168 first-half1000 repairs are admitted")
    return admission.RUNS / f"20261005-M0-null-MVC-indexed-array-46592261/replicate{index}/mz_null_indexed_mvc.json"


def validate_failure(index, failed, source, people, original):
    """Reuse existing indexed report validation, then prove exact1000 exhaustion."""
    path = failed_path(index)
    identity = IDENTITIES[index]
    if (str(source.get("job_id")) != identity["source_job_id"] or source.get("finished_utc") != identity["source_finished_utc"]
            or failed.get("job_id") != identity["failed_job_id"] or failed.get("finished_utc") != identity["failed_finished_utc"]
            or failed.get("status") != "technical_rival_failure" or failed.get("error") != NONCONVERGENCE
            or failed.get("source_baseline_validated") is not True or failed.get("baseline_lineage") is not None
            or failed.get("internal_seconds") != 8400 or failed.get("scored_primary_directions") != []
            or not isinstance(failed.get("fits"), list) or len(failed["fits"]) != 1):
        raise ValueError("Only this index's exact selected original worker and first-half1000 failure are admitted")
    # The outer worker uses runtime1500. Restore1000 only while checking the
    # immutable prior report with its existing validator; no fitting occurs.
    with patch.object(indexed, "RUNTIME_SWEEPS", 1000):
        admission.rival_identity(failed, "MVC", index, source, original, None)
        admission.menu.completed_support(failed)
        prior = admission.technical_rival(failed, "MVC", index, original, people)
        previous_runtime = indexed.runtime_specification(original)
    fit = failed["fits"][0]
    history = fit.get("objective_history")
    if (fit.get("converged") is not False or fit.get("subject_optimizer_failures") != []
            or not isinstance(history, list) or len(history) != 1000
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in history)):
        raise ValueError("Require exactly1000 finite sweeps without final subject optimizer failures")
    change = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if not math.isfinite(change) or not math.isfinite(threshold) or change <= threshold:
        raise ValueError("The returned first fit must exceed the unchanged convergence threshold")
    metadata = {
        "kind": "first_half_1000_sweep_exhaustion_fresh_both", "failed_report": str(path),
        **identity, "failed_started_utc": failed["started_utc"], "failed_error": failed["error"],
        "original_maximum_sweeps": 500, "previous_runtime_maximum_sweeps": 1000,
        "runtime_maximum_sweeps": RUNTIME_SWEEPS, "previous_runtime_implementation": previous_runtime,
        "previous_process_elapsed_seconds": prior["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"]], "previous_technical_attempt": prior,
        "final_objective_change": change, "unchanged_convergence_threshold": threshold,
        "completed_first_fit_available": False, "nonconverged_checkpoint_state_reused": False,
        "fresh_fit_directions": [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in indexed.HALVES],
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "qualifying_runtime1500_report": str(QUALIFYING_REPORT),
        "cost_accounting": "prior_failed_MVC_process_plus_new_MVC_process_once; shared_original_baseline_cost_not_added",
    }
    return metadata


def run(args):
    require_compute()
    path = failed_path(args.replicate_index)
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Keep the whole-process cap within1 through12600 seconds")
    output = args.output_dir.resolve()
    if any(output == failed_path(index).parent.resolve() or failed_path(index).parent.resolve() in output.parents
           for index in IDENTITIES):
        raise FileExistsError("Preserve the original1000 failure in place")
    original_source_validator, original_writer = indexed.validate_source, indexed.write_report
    metadata = None

    def validate_source(source, truth, people, index, original):
        nonlocal metadata
        if index != args.replicate_index:
            raise ValueError("The selected source index cannot change")
        lineage = original_source_validator(source, truth, people, index, original)
        if lineage is not None:
            raise ValueError("Use this index's unchanged original500 baseline, not a completion adapter")
        failed = json.loads(path.read_text())
        metadata = validate_failure(index, failed, source, people, original)
        return lineage

    def write_checkpoint(directory, report):
        report.update(
            phase=PHASE, repair_context_validated=metadata is not None, numerical_repair=deepcopy(metadata),
            failed_report=str(path), original_maximum_sweeps=500, previous_runtime_maximum_sweeps=1000,
            fitted_maximum_sweeps=RUNTIME_SWEEPS, wave_selected=False,
            wave_scope="single_indexed_MVC_numerical_repair", component_scope="MVC_only",
            retry_wave_selected=False, full_menu_wave_selected=False, retry_selected=False, automatic_retry=False,
            interpretation=f"Only null{args.replicate_index}'s diagnosed first-half1000 MVC exhaustion is repaired. Both MVC halves start fresh under1500; original500 M0/MZ scores are replayed and retained. No failed state or completed first fit is reused. No family, full-menu, rate, interval, early-kill or gate verdict.",
            cost_accounting="elapsed_seconds is this MVC process; cumulative adds the exact prior failed MVC process once, excluding shared baseline costs",
        )
        report["numerical_override"].update(
            qualifying_runtime1500_report=str(QUALIFYING_REPORT),
            qualification_boundary="Coordinator records runtime1500 first-convergence qualification and separately selects this diagnosed cohort; this worker never rereads pilots or selects a wave",
        )
        if metadata is not None and "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        original_writer(directory, report)

    # These bindings exist only in this one worker, with restoration on every
    # exit. The checked loop computes runtime1500 before timed source validation
    # and retains that local copy while the prior1000 report is checked.
    with ExitStack() as stack:
        for name, value in {"RUNTIME_SWEEPS": RUNTIME_SWEEPS, "MAX_SECONDS": MAX_SECONDS,
                            "REPORT_NAME": REPORT_NAME, "validate_source": validate_source,
                            "write_report": write_checkpoint}.items():
            stack.enter_context(patch.object(indexed, name, value))
        return indexed.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, choices=tuple(IDENTITIES), required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "replicate_index", "rival_diagnostics_complete",
        "repair_context_validated", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error") if key in report}, indent=2), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
