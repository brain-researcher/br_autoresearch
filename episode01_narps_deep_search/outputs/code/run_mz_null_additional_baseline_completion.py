#!/usr/bin/env python3
"""Complete only the missing original500 reverse MZ fit of null108 or null176.

Reuse the checked null35 completion engine through process-local identities.
The exact three-fit signal15 checkpoints, simulation and truth remain in place.
No baseline simulation, preserved-fit refit, numerical change or retry wave.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_baseline_completion as base


SOURCES = {
    108: ("46581418", "2026-10-05T07:27:38.777795+00:00"),
    176: ("46581754", "2026-10-05T07:27:47.062496+00:00"),
}
PHASE = "M0_null_MZ_additional_baseline_completion"
REPORT_NAME = "mz_null_additional_baseline_completion.json"
PRESERVED_DIRECTORIES = (
    base.baseline.RUNS / "20261005-M0-null-baseline-array-46579924",
    base.baseline.RUNS / "20261005-M0-null-baseline-array-46580387",
    base.baseline.RUNS / "20261005-M0-null-baseline35-completion-46586603",
)


def source_directory(index):
    if type(index) is not int or index not in SOURCES:
        raise ValueError("Only the two diagnosed null baseline indices108 and176 are selected")
    return base.baseline.source_directory(index, 46579924, 46580387)


@contextmanager
def selected_source(index):
    directory = source_directory(index)
    worker, finished = SOURCES[index]
    values = {"REPLICATE_INDEX": index, "SEED": base.indexed.null_seed(index), "SOURCE": directory,
              "SOURCE_JOB_ID": worker, "SOURCE_FINISHED_UTC": finished,
              "PHASE": PHASE, "REPORT_NAME": REPORT_NAME}
    with ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(patch.object(base, name, value))
        yield


def completion_context(index, source, truth, people, spec):
    """Expose exact selected-source validation without weakening the old engine."""
    with selected_source(index):
        return base.completion_context(source, truth, people, spec)


def run(args):
    require_compute()
    source_directory(args.replicate_index)
    output = args.output_dir.resolve()
    if any(output == path.resolve() or path.resolve() in output.parents for path in PRESERVED_DIRECTORIES):
        raise FileExistsError("Preserve both original baseline arrays and the completed null35 repair")
    original_writer = base.write_report

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, selected_replicates=[args.replicate_index],
                      early_kill="not_evaluated", automatic_retry=False,
                      interpretation=f"Only null{args.replicate_index}'s missing reverse MZ fit is computed under original500. Its three preserved fits are validated and rescored; simulation and truth remain at the original source. No family, full-menu selection, rate, interval, early-kill or gate verdict is supplied.")
        original_writer(directory, report)

    # The original engine owns compute/CPU/spec/cap/fresh-output guards, timed
    # source reads and validation, exact prefix reuse, rescoring and one fit.
    # Restore all references on success, technical failure or a guard exception.
    with selected_source(args.replicate_index), patch.object(base, "write_report", write_checkpoint):
        return base.run(argparse.Namespace(implementation=args.implementation,
            output_dir=output, max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, choices=tuple(SOURCES), required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=2700)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "replicate_index", "seed", "source_job_id",
          "pairwise_diagnostic_complete", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error")
          if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["pairwise_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
