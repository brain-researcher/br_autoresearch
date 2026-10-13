#!/usr/bin/env python3
"""Target-free tests for the EP10 observed-search production launcher."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS_DIR))

import launch_v3_m2_observed_search_v1 as launch  # noqa: E402


def _arguments(prefix: Path) -> list[str]:
    return [
        "--interface-root",
        str(prefix / "interface"),
        "--geometry-root",
        str(prefix / "geometry"),
        "--target-root",
        str(prefix / "target"),
        "--m1-checkpoint-root",
        str(prefix / "m1-checkpoints"),
        "--core-report",
        str(prefix / "core.json"),
        "--actual-k-report",
        str(prefix / "actual-k.json"),
        "--m1-adequacy-report",
        str(prefix / "m1-adequacy.json"),
        "--optional-report",
        str(prefix / "optional.json"),
        "--scale-report",
        str(prefix / "scale.json"),
        "--falsifier-timing-report",
        str(prefix / "falsifier-timing.json"),
    ]


class _Bundle:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.values = {
            "cells": object(),
            "moments": object(),
            "candidates": object(),
            "geometry": object(),
            "pairs": object(),
            "position_strata": object(),
        }
        for name, value in self.values.items():
            setattr(self, name, value)

    def runner_kwargs(self) -> dict[str, object]:
        self._calls.append("bundle_kwargs")
        return dict(self.values)


class ObservedLaunchTests(unittest.TestCase):
    def test_validation_failure_prevents_every_data_bearing_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prefix = Path(raw)
            args = launch.build_parser().parse_args(_arguments(prefix))
            failure = RuntimeError("prerequisite failed")
            with (
                mock.patch.object(
                    launch.runner,
                    "validate_prerequisite_reports",
                    side_effect=failure,
                ) as validate,
                mock.patch.object(
                    launch.executor.FrozenExecutionInventory, "from_tsv"
                ) as inventory,
                mock.patch.object(
                    launch.loader, "load_observed_inputs_from_namespace"
                ) as load_inputs,
                mock.patch.object(
                    launch.loader, "import_outer_m1_for_observed"
                ) as import_m1,
                mock.patch.object(launch.runner, "run_observed_search") as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "prerequisite failed"):
                    launch.launch_from_namespace(args)

            validate.assert_called_once()
            supplied = validate.call_args.args[0]
            self.assertEqual(
                tuple(supplied),
                (
                    "core",
                    "actual_k",
                    "m1_adequacy",
                    "optional",
                    "scale",
                    "falsifier_timing",
                ),
            )
            self.assertEqual(supplied["core"], prefix / "core.json")
            inventory.assert_not_called()
            load_inputs.assert_not_called()
            import_m1.assert_not_called()
            run.assert_not_called()

    def test_exact_order_inventory_reuse_forwarding_and_derived_output(self) -> None:
        calls: list[str] = []
        inventory_object = object()
        offsets = object()
        coordinator_object = object()
        payloads = {name: {"status": "pass"} for name, _, _ in launch.REPORT_ARGUMENTS}
        validated = SimpleNamespace(payloads=payloads)
        result = SimpleNamespace(
            summary_path=Path("/aggregate/terminal_summary.json"),
            manifest_path=Path("/aggregate/terminal_manifest.json"),
        )

        with tempfile.TemporaryDirectory() as raw:
            scratch = Path(raw) / "scratch"
            scratch.mkdir()
            prefix = Path(raw) / "arguments"
            bundle = _Bundle(calls)

            def validate(reports: Any) -> Any:
                calls.append("validate")
                self.assertEqual(set(reports), set(payloads))
                return validated

            def inventory() -> object:
                calls.append("inventory")
                return inventory_object

            def load_inputs(args: Any) -> _Bundle:
                calls.append("load")
                self.assertEqual(args.target_root, prefix / "target")
                return bundle

            def import_m1(
                seen_bundle: object,
                checkpoint_root: Path,
                *,
                execution_inventory: object,
            ) -> Any:
                calls.append("import")
                self.assertIs(seen_bundle, bundle)
                self.assertEqual(checkpoint_root, prefix / "m1-checkpoints")
                self.assertIs(execution_inventory, inventory_object)
                return SimpleNamespace(offsets=offsets)

            def coordinator(output_root: Path) -> object:
                calls.append("coordinator")
                self.assertEqual(
                    output_root,
                    scratch
                    / "br_autoresearch"
                    / "episode10_single_cell_coprojection"
                    / "v3_m2_observed_search_v1_job_7654321",
                )
                return coordinator_object

            def run(**kwargs: Any) -> Any:
                calls.append("run")
                self.assertIs(kwargs["execution_inventory"], inventory_object)
                self.assertIs(kwargs["fixed_m1_offsets"], offsets)
                self.assertIs(kwargs["prerequisite_reports"], payloads)
                self.assertEqual(kwargs["job_id"], "7654321")
                self.assertEqual(
                    kwargs["output_root"],
                    scratch
                    / "br_autoresearch"
                    / "episode10_single_cell_coprojection"
                    / "v3_m2_observed_search_v1_job_7654321",
                )
                self.assertIs(
                    kwargs["postselection_callback"], coordinator_object
                )
                for name, value in bundle.values.items():
                    self.assertIs(kwargs[name], value)
                return result

            with (
                mock.patch.dict(
                    os.environ,
                    {"SLURM_JOB_ID": "7654321", "SCRATCH": str(scratch)},
                    clear=True,
                ),
                mock.patch.object(
                    launch.runner,
                    "validate_prerequisite_reports",
                    side_effect=validate,
                ),
                mock.patch.object(
                    launch.executor.FrozenExecutionInventory,
                    "from_tsv",
                    side_effect=inventory,
                ),
                mock.patch.object(
                    launch.loader,
                    "load_observed_inputs_from_namespace",
                    side_effect=load_inputs,
                ),
                mock.patch.object(
                    launch.loader,
                    "import_outer_m1_for_observed",
                    side_effect=import_m1,
                ),
                mock.patch.object(
                    launch.selected_refit,
                    "SelectedRefitCoordinator",
                    side_effect=coordinator,
                ),
                mock.patch.object(
                    launch.runner, "run_observed_search", side_effect=run
                ),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    status = launch.main(_arguments(prefix))

        self.assertEqual(status, 0)
        self.assertEqual(
            calls,
            [
                "validate",
                "inventory",
                "load",
                "import",
                "coordinator",
                "bundle_kwargs",
                "run",
            ],
        )
        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                "/aggregate/terminal_summary.json",
                "/aggregate/terminal_manifest.json",
            ],
        )

    def test_missing_live_environment_fails_after_validation_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = launch.build_parser().parse_args(_arguments(Path(raw)))
            validated = SimpleNamespace(payloads={})
            cases = (
                ({"SCRATCH": raw}, "SLURM_JOB_ID"),
                ({"SLURM_JOB_ID": "not-numeric", "SCRATCH": raw}, "SLURM_JOB_ID"),
                ({"SLURM_JOB_ID": "12345"}, "SCRATCH"),
            )
            for environment, message in cases:
                with self.subTest(environment=environment), mock.patch.dict(
                    os.environ, environment, clear=True
                ), mock.patch.object(
                    launch.runner,
                    "validate_prerequisite_reports",
                    return_value=validated,
                ) as validate, mock.patch.object(
                    launch.executor.FrozenExecutionInventory, "from_tsv"
                ) as inventory, mock.patch.object(
                    launch.loader, "load_observed_inputs_from_namespace"
                ) as load_inputs:
                    with self.assertRaisesRegex(launch.ObservedLaunchError, message):
                        launch.launch_from_namespace(args)
                    validate.assert_called_once()
                    inventory.assert_not_called()
                    load_inputs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
