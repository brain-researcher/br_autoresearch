"""Run one preplanned EP12 adaptive development configuration.

This entry point is deliberately separate from the coverage CLI so downstream
Slurm arrays consume the frozen successor configurations without changing an
already-running coverage array.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .adaptive_search import validate_development_configuration
from .development_trial import run_development_trial
from .policy import EpisodePolicy, digest_object
from .runtime import EPISODE_ROOT, SCRATCH_ROOT


def load_successor_configuration(
    *,
    configuration_path: Path,
    index: int,
) -> dict[str, object]:
    configuration_path = configuration_path.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    if configuration_path != allowed_output and allowed_output not in configuration_path.parents:
        raise ValueError("successor configuration manifest must stay under EP12 outputs")
    payload = json.loads(configuration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("configurations"), list):
        raise ValueError("successor configuration manifest has the wrong schema")
    configurations = payload["configurations"]
    if index < 0 or index >= len(configurations):
        raise IndexError(f"successor index {index} is outside 0..{len(configurations) - 1}")
    configuration = dict(configurations[index])
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    validate_development_configuration(policy, configuration)
    recorded = payload.get("configuration_sha256", [])
    if len(recorded) != len(configurations):
        raise ValueError("successor configuration hash list is incomplete")
    if str(recorded[index]) != digest_object(configuration):
        raise ValueError("successor configuration hash mismatch")
    return configuration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-adaptive-trial")
    parser.add_argument("--configuration-path", type=Path, required=True)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRATCH_ROOT / "development_snapshot_v1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    configuration = load_successor_configuration(
        configuration_path=arguments.configuration_path,
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
                "technical_failure_count": report["aggregate"][
                    "technical_failure_count"
                ],
                "final_connectivity_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if not report["aggregate"]["technical_failure_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
