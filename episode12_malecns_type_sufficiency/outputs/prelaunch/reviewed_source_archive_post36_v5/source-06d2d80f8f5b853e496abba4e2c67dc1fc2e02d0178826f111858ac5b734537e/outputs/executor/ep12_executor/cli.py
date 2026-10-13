"""Command-line entry point for the synthetic-only EP12 executor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .synthetic import (
    EPISODE_ROOT,
    SCRATCH_ROOT,
    make_controller,
    qualify_all,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ep12-executor",
        description=(
            "Fail-closed EP12 executor for synthetic qualification and authorized "
            "development-only real execution; final connectivity remains disabled."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    qualify = subcommands.add_parser(
        "qualify-synthetic",
        help="run the complete synthetic control-plane qualification matrix",
    )
    qualify.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    qualify.add_argument("--scratch-root", type=Path, default=SCRATCH_ROOT)
    qualify.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "executor_qualification",
    )

    methods = subcommands.add_parser(
        "qualify-methods",
        help="run the actual T/U/M fitter against generated biological counterexamples",
    )
    methods.add_argument(
        "--tier", choices=("smoke", "qualification", "stress"), default="smoke"
    )
    methods.add_argument("--repetitions", type=int)
    methods.add_argument("--whole-types", type=int)
    methods.add_argument("--integration-draws", type=int)
    methods.add_argument("--folds", type=int)
    methods.add_argument("--bootstrap-replicates", type=int)
    methods.add_argument("--output-dir", type=Path)
    methods.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    methods.add_argument("--policy", type=Path)
    methods.add_argument("--contract", type=Path, required=True)
    methods.add_argument("--qualification-spec", type=Path, required=True)
    methods.add_argument("--transition-manifest", type=Path, required=True)
    methods.add_argument("--transition-manifest-sha256", required=True)
    methods.add_argument("--transition-receipt-sha256", required=True)

    preflight = subcommands.add_parser(
        "preflight-source",
        help="authenticate Arrow schemas and freeze annotation-only whole-type roles",
    )
    preflight.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "prelaunch" / "source_preflight",
    )

    margin = subcommands.add_parser(
        "diagnose-margin",
        help="test fixed-margin compatibility on generated grouped counterexamples",
    )
    margin.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "prelaunch" / "margin_compatibility",
    )

    development = subcommands.add_parser(
        "materialize-development",
        help="open and materialize only development-role MaleCNS focal connectivity",
    )
    development.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "development_run001" / "materialization",
    )
    development.add_argument(
        "--scratch-dir",
        type=Path,
        default=SCRATCH_ROOT / "development_snapshot_v1",
    )

    hierarchy = subcommands.add_parser(
        "materialize-partner-hierarchy",
        help="freeze annotation-only partner parents for the hierarchical vocabulary",
    )
    hierarchy.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "development_run002" / "preparation",
    )
    hierarchy.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRATCH_ROOT / "development_snapshot_v1",
    )

    plan = subcommands.add_parser(
        "prepare-coverage-plan",
        help="freeze the executable 18-row real development covering array",
    )
    plan.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "development_run002" / "search_plan",
    )

    trial = subcommands.add_parser(
        "run-development-trial",
        help="run a complete real T/U/M triplet trial over development-role types",
    )
    trial.add_argument(
        "--output-dir",
        type=Path,
        default=EPISODE_ROOT / "outputs" / "development_run001" / "trial001",
    )
    trial.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRATCH_ROOT / "development_snapshot_v1",
    )
    trial.add_argument("--config-manifest", type=Path)
    trial.add_argument("--config-index", type=int)

    reconcile = subcommands.add_parser(
        "reconcile-development-trial",
        help="reclassify snapshot-verified zero-count failures without changing scores",
    )
    reconcile.add_argument("--source-trial-dir", type=Path, required=True)
    reconcile.add_argument("--output-dir", type=Path, required=True)
    reconcile.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRATCH_ROOT / "development_snapshot_v1",
    )

    for name in ("status", "replay"):
        command = subcommands.add_parser(name, help=f"{name} a synthetic run journal")
        command.add_argument("run_dir", type=Path)
        command.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "qualify-synthetic":
        report = qualify_all(
            episode_root=arguments.episode_root,
            scratch_root=arguments.scratch_root,
            output_dir=arguments.output_dir,
        )
        print(
            json.dumps(
                {
                    "qualification_passed": report["qualification_passed"],
                    "scope": report["qualification_scope"],
                    "output_dir": str(arguments.output_dir.resolve()),
                    "real_data_execution_enabled": False,
                },
                sort_keys=True,
            )
        )
        return 0 if report["qualification_passed"] else 1

    if arguments.command == "qualify-methods":
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        from .scientific_qualification import run_scientific_qualification

        output_dir = arguments.output_dir or (
            EPISODE_ROOT
            / "outputs"
            / "prelaunch"
            / f"scientific_qualification_{arguments.tier}"
        )
        report = run_scientific_qualification(
            tier=arguments.tier,
            output_dir=output_dir,
            repetitions=arguments.repetitions,
            whole_types=arguments.whole_types,
            integration_draws=arguments.integration_draws,
            folds=arguments.folds,
            bootstrap_replicates=arguments.bootstrap_replicates,
            contract_path=arguments.contract,
            qualification_spec_path=arguments.qualification_spec,
            transition_manifest_path=arguments.transition_manifest,
            transition_manifest_sha256=arguments.transition_manifest_sha256,
            transition_receipt_sha256=arguments.transition_receipt_sha256,
            episode_root=arguments.episode_root,
            policy_path=arguments.policy,
        )
        print(
            json.dumps(
                {
                    "qualification_passed": report["qualification_passed"],
                    "scope": report["qualification_scope"],
                    "output_dir": str(output_dir.resolve()),
                    "real_connectivity_accessed": False,
                    "cost_gate_passed": report["cost_profile"]["cost_gate_passed"],
                },
                sort_keys=True,
            )
        )
        return 0 if report["qualification_passed"] else 1

    if arguments.command == "preflight-source":
        from .preflight import run_source_preflight

        report = run_source_preflight(output_dir=arguments.output_dir)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["preflight_passed"] else 1

    if arguments.command == "diagnose-margin":
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        from .margin_diagnostic import run_margin_compatibility_diagnostic

        report = run_margin_compatibility_diagnostic(output_dir=arguments.output_dir)
        print(
            json.dumps(
                {
                    "diagnostic_complete": True,
                    "output_dir": str(arguments.output_dir.resolve()),
                    "fixed_margin": report["fixed_margin_nats_per_observed_count"],
                    "maximum_observed_direction_delta": report[
                        "maximum_observed_direction_delta"
                    ],
                    "real_connectivity_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "materialize-development":
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        from .development_data import materialize_development_view

        report = materialize_development_view(
            output_dir=arguments.output_dir,
            scratch_dir=arguments.scratch_dir,
        )
        print(
            json.dumps(
                {
                    "development_execution_started": True,
                    "output_dir": str(arguments.output_dir.resolve()),
                    "scratch_dir": str(arguments.scratch_dir.resolve()),
                    "development_type_count": report["development_type_count"],
                    "final_type_count_locked": report["final_type_count_locked"],
                    "final_focal_connectivity_materialized_or_summarized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "materialize-partner-hierarchy":
        from .development_data import materialize_partner_hierarchy

        report = materialize_partner_hierarchy(
            output_dir=arguments.output_dir,
            snapshot_dir=arguments.snapshot_dir,
        )
        print(json.dumps(report, sort_keys=True))
        return 0

    if arguments.command == "prepare-coverage-plan":
        from .adaptive_search import prepare_coverage_plan

        report = prepare_coverage_plan(output_dir=arguments.output_dir)
        print(json.dumps(report, sort_keys=True))
        return 0

    if arguments.command == "run-development-trial":
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        from .adaptive_search import load_coverage_configuration
        from .development_trial import run_development_trial

        if (arguments.config_manifest is None) != (arguments.config_index is None):
            raise SystemExit("--config-manifest and --config-index must be provided together")
        configuration = None
        if arguments.config_manifest is not None:
            configuration = load_coverage_configuration(
                plan_dir=arguments.config_manifest,
                index=arguments.config_index,
            )

        report = run_development_trial(
            output_dir=arguments.output_dir,
            snapshot_dir=arguments.snapshot_dir,
            configuration=configuration,
        )
        print(
            json.dumps(
                {
                    "trial_id": report["trial_id"],
                    "status": report["status"],
                    "data_role": "development_only",
                    "valid_type_count": report["aggregate"]["valid_type_count"],
                    "final_connectivity_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "completed" else 1

    if arguments.command == "reconcile-development-trial":
        from .development_trial import reconcile_zero_count_failures

        report = reconcile_zero_count_failures(
            source_trial_dir=arguments.source_trial_dir,
            output_dir=arguments.output_dir,
            snapshot_dir=arguments.snapshot_dir,
        )
        print(
            json.dumps(
                {
                    "trial_id": report["trial_id"],
                    "status": report["status"],
                    "valid_type_count": report["aggregate"]["valid_type_count"],
                    "unscorable_type_count": report["aggregate"][
                        "unscorable_type_count"
                    ],
                    "technical_failure_count": report["aggregate"][
                        "technical_failure_count"
                    ],
                    "scores_refit_or_changed": False,
                    "final_connectivity_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if not report["aggregate"]["technical_failure_count"] else 1

    run_dir = arguments.run_dir.resolve()
    if not (run_dir / "trial_ledger.jsonl").is_file():
        raise SystemExit(f"synthetic run journal does not exist: {run_dir / 'trial_ledger.jsonl'}")
    controller = make_controller(
        run_dir,
        (arguments.episode_root / "SEARCH_POLICY.yaml").resolve(),
    )
    if arguments.command == "status":
        print(json.dumps(controller.status(), indent=2, sort_keys=True))
    elif arguments.command == "replay":
        state = controller.replay()
        print(
            json.dumps(
                {
                    "status": controller.status(),
                    "event_ids": [record["event_id"] for record in state.records],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0
