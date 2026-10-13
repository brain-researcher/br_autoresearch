#!/usr/bin/env python3
"""Launch the sealed EP10 v3 observed M2 search inside one live Slurm job.

The six aggregate prerequisites are validated before this entry point opens
development membership or imports an outer-M1 checkpoint.  Output is always
claimed beneath the current job's EP10 Scratch directory; there is no caller-
selectable output path.  The terminal selected full-development refit runs in
the same live process so its nonserializable session authority cannot be lost.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import load_v3_m2_observed_inputs_v1 as loader  # noqa: E402
import run_v3_m2_observed_search_v1 as runner  # noqa: E402
import v3_m2_selected_refit_v1 as selected_refit  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402


EPISODE_SCRATCH_COMPONENTS = (
    "br_autoresearch",
    "episode10_single_cell_coprojection",
)
REPORT_ARGUMENTS = (
    ("core", "core_report", "--core-report"),
    ("actual_k", "actual_k_report", "--actual-k-report"),
    ("m1_adequacy", "m1_adequacy_report", "--m1-adequacy-report"),
    ("optional", "optional_report", "--optional-report"),
    ("scale", "scale_report", "--scale-report"),
    (
        "falsifier_timing",
        "falsifier_timing_report",
        "--falsifier-timing-report",
    ),
)


class ObservedLaunchError(RuntimeError):
    """The top-level launch is not inside the required live Slurm context."""


def build_parser() -> argparse.ArgumentParser:
    """Build the production parser with four sealed roots and six reports."""

    parser = argparse.ArgumentParser(description=__doc__)
    loader.add_observed_input_arguments(parser)
    for _, destination, flag in REPORT_ARGUMENTS:
        parser.add_argument(flag, dest=destination, type=Path, required=True)
    return parser


def _report_paths(args: argparse.Namespace) -> Mapping[str, Path]:
    return {
        report_name: getattr(args, destination)
        for report_name, destination, _ in REPORT_ARGUMENTS
    }


def _live_slurm_context() -> tuple[str, Path]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ObservedLaunchError(
            "observed launch requires a numeric live SLURM_JOB_ID"
        )
    scratch_value = os.environ.get("SCRATCH", "")
    if not scratch_value:
        raise ObservedLaunchError("observed launch requires SCRATCH")
    try:
        scratch_root = Path(scratch_value).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ObservedLaunchError("SCRATCH is unavailable") from error
    if not scratch_root.is_dir():
        raise ObservedLaunchError("SCRATCH is not a directory")
    return job_id, scratch_root


def launch_from_namespace(args: argparse.Namespace) -> Any:
    """Validate first, then open only the sealed development launch inputs."""

    validated = runner.validate_prerequisite_reports(_report_paths(args))
    job_id, scratch_root = _live_slurm_context()

    execution_inventory = executor.FrozenExecutionInventory.from_tsv()
    bundle = loader.load_observed_inputs_from_namespace(args)
    imported_m1 = loader.import_outer_m1_for_observed(
        bundle,
        args.m1_checkpoint_root,
        execution_inventory=execution_inventory,
    )
    output_root = scratch_root.joinpath(
        *EPISODE_SCRATCH_COMPONENTS,
        f"{runner.OUTPUT_PREFIX}{job_id}",
    )
    refit_coordinator = selected_refit.SelectedRefitCoordinator(
        output_root,
    )
    return runner.run_observed_search(
        **bundle.runner_kwargs(),
        fixed_m1_offsets=imported_m1.offsets,
        prerequisite_reports=validated.payloads,
        output_root=output_root,
        job_id=job_id,
        execution_inventory=execution_inventory,
        postselection_callback=refit_coordinator,
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = launch_from_namespace(build_parser().parse_args(argv))
    print(result.summary_path)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ObservedLaunchError",
    "build_parser",
    "launch_from_namespace",
    "main",
)
