#!/usr/bin/env python3
"""Fit one rival on null108/176 after its exact baseline completion46591918.

Keep truth/events at the original failed worker. Reuse the checked completion
validator and scientific runners through process-local bindings only: original
500 for MC/MV/MVZ, original500 baseline plus two fresh1000 fits for MVC.
Execution requires the coordinator's separate terminal predecessor selection.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_additional_baseline_completion as additional
import run_mz_null_completed_baseline_rival as completed
import run_mz_null_indexed_mvc as mvc


null = completed.null
MODELS = ("MC", "MV", "MVZ", "MVC")
MAX_SECONDS = {"MC": 2400, "MV": 2400, "MVZ": 3300, "MVC": 8400}
COMPLETION_JOB_ID = "46591918"
PHASE = "M0_null_additional_completed_baseline_rival_diagnostics"
REPORT_NAME = "mz_null_additional_completed_rival.json"
COMPLETION_DIRECTORIES = {
    index: additional.base.baseline.RUNS / f"20261005-M0-null-baseline{index}-completion-{COMPLETION_JOB_ID}"
    for index in additional.SOURCES
}
PRESERVED_DIRECTORIES = (*mvc.PRESERVED_DIRECTORIES, *additional.PRESERVED_DIRECTORIES,
                         *COMPLETION_DIRECTORIES.values())


def source_paths(index):
    directory = additional.source_directory(index)
    return directory, COMPLETION_DIRECTORIES[index] / additional.REPORT_NAME


@contextmanager
def selected_baseline(index):
    directory, report = source_paths(index)
    # This is the exact identity context used by additional.completion_context;
    # the old validator calls that same unchanged completion engine within it.
    with additional.selected_source(index), ExitStack() as stack:
        for name, value in {
            "COMPLETION_JOB_ID": COMPLETION_JOB_ID,
            "COMPLETION_DIRECTORY": report.parent,
            "COMPLETION_REPORT": report,
            "FAILED_REPORT": directory / additional.base.SOURCE_REPORT,
        }.items():
            stack.enter_context(patch.object(completed, name, value))
        yield


def validate_completed_baseline(index, report, failed, truth, people, original):
    """Admit the distinct completed phase, retaining exact three-fit histories/costs."""
    with selected_baseline(index):
        lineage = completed.validate_completed_baseline(report, failed, truth, people, original)
    if report.get("automatic_retry") is not False:
        raise ValueError("The additional baseline completion selects no automatic retry")
    return lineage


def run(args):
    require_compute()
    directory, completion_report = source_paths(args.replicate_index)
    if args.model not in MODELS:
        raise ValueError("Select only MC, MV, MVZ or MVC on completed null108/176")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS[args.model]:
        raise ValueError("Keep MC/MV2400, MVZ3300 or MVC8400 whole-process bounds")
    output = args.output_dir.resolve()
    if any(output == path.resolve() or path.resolve() in output.parents for path in PRESERVED_DIRECTORIES):
        raise FileExistsError("Preserve the original arrays, completed baselines and qualifying pilots")
    engine = mvc if args.model == "MVC" else null
    original_writer = engine.write_report
    lineage = None

    def validate_source(report, truth, people, original):
        nonlocal lineage
        # Called only inside the selected checked runner's whole-process timer.
        failed = json.loads((directory / additional.base.SOURCE_REPORT).read_text())
        lineage = validate_completed_baseline(args.replicate_index, report, failed, truth, people, original)
        return lineage

    def validate_mvc_source(report, truth, people, index, original):
        if index != args.replicate_index:
            raise ValueError("The selected completed baseline index cannot change")
        return validate_source(report, truth, people, original)

    def write_checkpoint(destination, report):
        report.update(
            phase=PHASE, source_job_id=COMPLETION_JOB_ID if lineage is not None else None,
            source_baseline_validated=lineage is not None, baseline_lineage=lineage,
            selected_replicates=[args.replicate_index], selected_models=[args.model],
            fixed_cohort_count=200, component_scope=f"{args.model}_only",
            wave_selected=False, wave_scope="single_completed_baseline_component",
            retry_selected=False, retry_wave_selected=False, full_menu_wave_selected=False,
            automatic_retry=False, simulation_copied=False, early_kill="not_evaluated",
            baseline_validation_maximum_sweeps=500,
            fitted_maximum_sweeps=1000 if args.model == "MVC" else 500,
            interpretation=f"One {args.model} component on null{args.replicate_index} after its exact baseline completion46591918. Original500 M0/MZ scores are replayed and retained; only this rival is freshly fitted in both halves. No family, full-menu selection, rate, interval, gate or early-kill verdict is supplied.",
            cost_accounting="elapsed_seconds is this rival process only. baseline_lineage references shared original/completion costs; those costs are not added to each rival process.",
            process_state_boundary="The coordinator verifies terminal successful baseline completion46591918 and no competing valid-source fit before execution. Old components bound to immutable failed original baselines reject before fitting and need not finish first; report fields are not scheduler evidence.",
        )
        original_writer(destination, report)

    runtime_args = argparse.Namespace(**{**vars(args), "source_dir": directory, "output_dir": output})
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, "REPORT_NAME", REPORT_NAME))
        stack.enter_context(patch.object(engine, "write_report", write_checkpoint))
        if args.model == "MVC":
            bindings = {"source_paths": source_paths, "validate_source": validate_mvc_source,
                        "PHASE": PHASE, "PRESERVED_DIRECTORIES": PRESERVED_DIRECTORIES}
        else:
            bindings = {"SOURCE": directory, "SOURCE_REPORT": str(completion_report),
                        "SOURCE_JOB_ID": None, "REPLICATE_INDEX": args.replicate_index,
                        "SEED": additional.base.indexed.null_seed(args.replicate_index),
                        "validate_source": validate_source}
        for name, value in bindings.items():
            stack.enter_context(patch.object(engine, name, value))
        return engine.run(runtime_args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, choices=tuple(additional.SOURCES), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "fitted_model", "replicate_index", "seed",
          "source_job_id", "source_baseline_validated", "rival_diagnostics_complete", "diagnostics", "elapsed_seconds", "error")
          if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
