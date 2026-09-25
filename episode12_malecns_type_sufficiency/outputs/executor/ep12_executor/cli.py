"""Command-line entry point for the synthetic-only EP12 executor."""

from __future__ import annotations

import argparse
import json
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
        description="Fail-closed EP12 executor; only synthetic qualification is enabled.",
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
