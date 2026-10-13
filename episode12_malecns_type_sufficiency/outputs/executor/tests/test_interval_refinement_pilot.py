from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from ep12_executor import interval_refinement_pilot as pilot

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


SLURM_ROOT = Path(__file__).resolve().parents[1] / "slurm"


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class IntervalRefinementPilotTests(unittest.TestCase):
    @staticmethod
    def _root(temporary: str) -> tuple[Path, Path]:
        root = Path(temporary) / "episode12"
        output = (
            root
            / "outputs"
            / "scientific_qualification"
            / "pilots"
            / pilot.PILOT_ID
        )
        return root, output

    @staticmethod
    def _policy(margin: float = 0.02) -> SimpleNamespace:
        return SimpleNamespace(
            meaningful_margin=margin,
            policy_hash="a" * 64,
        )

    @staticmethod
    def _generated_record(
        cell: object,
        *,
        replicate_index: int,
        whole_types: int,
        margin: float,
        bootstrap_replicates: int,
        integration_draws: int,
        folds: int,
    ) -> dict[str, object]:
        assert whole_types == 8
        assert margin == 0.02
        assert bootstrap_replicates == 1_999
        assert integration_draws in {512, 1_024, 2_048}
        assert folds == 2
        offset = (replicate_index - 35.5) / 10_000.0
        estimate = [offset, -offset, 0.0]
        return {
            "cell_id": cell.cell_id,
            "scenario": cell.scenario,
            "truth_family": pilot.SCENARIO_TRUTH[cell.scenario],
            "replicate_index": replicate_index,
            "terminal": "unresolved",
            "intervals": {
                "method": "paired_whole_type_studentized_bootstrap_max_t",
                "confidence_level": 0.95,
                "labels": ["left_to_right", "right_to_left", "aggregate"],
                "estimate": estimate,
                "standard_error": [0.01, 0.01, 0.005],
                "critical_max_t": 2.0,
                "lower": [value - 0.1 for value in estimate],
                "upper": [value + 0.1 for value in estimate],
                "bootstrap_replicates": 1_999,
                "bootstrap_zero_standard_error_draws": 0,
                "variable_coordinate_count": 3,
                "directions_resampled_together": True,
                "unit": "whole_provider_type",
                "margin": 0.02,
            },
            "controls": {
                "endpoint_side_permutation_p": 1.0,
                "endpoint_mass_control_pass": True,
                "leave_one_type_out_directional_gain": [
                    [offset, -offset] for _ in range(whole_types)
                ],
                "influence_control_pass": True,
                "all_pass": True,
            },
            "numerical_failures": [],
            "type_records": [
                {
                    "synthetic_type_index": index,
                    "deltas": [offset, -offset],
                    "source_fit_metadata": {},
                }
                for index in range(whole_types)
            ],
            "wall_seconds": 0.1,
            "cpu_seconds": 0.1,
        }

    def test_predeclared_inventory_has_twelve_cells_and_twenty_four_tasks(self) -> None:
        self.assertEqual(len(pilot.PILOT_CELLS), 12)
        central = [
            cell
            for cell in pilot.PILOT_CELLS
            if (
                cell.neurons_per_side,
                cell.count_total,
                cell.strength,
                cell.missing_fraction,
            )
            == (7, 100, 1.8, 0.1)
        ]
        self.assertEqual({cell.scenario for cell in central}, set(pilot.SCENARIO_TRUTH))
        edges = [cell for cell in pilot.PILOT_CELLS if cell not in central]
        self.assertEqual(
            {cell.scenario for cell in edges},
            {
                "side_specific_endpoint_loss",
                "anatomy_quality_confound",
                "high_strength_or_influential_cells",
            },
        )
        self.assertTrue(
            all(
                (
                    cell.neurons_per_side,
                    cell.count_total,
                    cell.strength,
                    cell.missing_fraction,
                )
                == (5, 10, 2.2, 0.0)
                for cell in edges
            )
        )
        definitions = [pilot.task_definition(index) for index in range(24)]
        self.assertEqual({item[0] for item in definitions}, {0, 1})
        self.assertEqual(len({(item[0], item[2].base_cell_id) for item in definitions}), 24)
        with self.assertRaises(ValueError):
            pilot.task_definition(24)

    def test_shard_uses_disjoint_block_seed_namespace_and_refuses_overwrite(self) -> None:
        captured: list[str] = []

        def generated(*args: object, **kwargs: object) -> dict[str, object]:
            captured.append(args[0].cell_id)
            return self._generated_record(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy()
            ), mock.patch.object(
                pilot, "run_generated_replicate", side_effect=generated
            ):
                first = pilot.run_shard(0, output_root=output, episode_root=root)
                second = pilot.run_shard(12, output_root=output, episode_root=root)
                with self.assertRaises(FileExistsError):
                    pilot.run_shard(0, output_root=output, episode_root=root)

        self.assertEqual(len(first["records"]), 72)
        self.assertEqual(len(second["records"]), 72)
        self.assertTrue(all(pilot.PILOT_ID in value for value in captured))
        self.assertTrue(all("block_01" in value for value in captured[:72]))
        self.assertTrue(all("block_02" in value for value in captured[72:]))
        self.assertNotEqual(captured[0], captured[72])
        self.assertFalse(first["scientific_gate_applied"])
        self.assertFalse(first["real_connectivity_accessed"])

    def test_policy_margin_and_output_boundary_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy(0.021)
            ), mock.patch.object(pilot, "run_generated_replicate") as generated:
                with self.assertRaisesRegex(ValueError, "meaningful margin"):
                    pilot.run_shard(0, output_root=output, episode_root=root)
                generated.assert_not_called()
            with self.assertRaisesRegex(ValueError, "must remain"):
                pilot.run_shard(
                    0,
                    output_root=root / "outputs" / "somewhere-else",
                    episode_root=root,
                )

    def test_interval_numerical_failure_is_retained_for_aggregation(self) -> None:
        cell = pilot.QualificationCell(
            cell_id="interval_refinement_v1__block_01__test",
            scenario="adequate_template",
            neurons_per_side=7,
            count_total=100,
            strength=1.8,
            missing_fraction=0.1,
            acceptance_role="interval_refinement_pilot_only",
        )
        with mock.patch.object(
            pilot,
            "run_generated_replicate",
            side_effect=pilot.ScientificModelError("nonfinite critical"),
        ):
            record = pilot._run_pilot_replicate(
                cell, replicate_index=4, margin=0.02
            )
        self.assertEqual(record["terminal"], "numerical_failure")
        self.assertEqual(record["replicate_index"], 4)
        self.assertIn("nonfinite critical", record["numerical_failures"][0])

    def test_aggregate_verifies_all_records_and_remains_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy()
            ), mock.patch.object(
                pilot,
                "run_generated_replicate",
                side_effect=self._generated_record,
            ):
                for task_id in range(pilot.SHARD_COUNT):
                    pilot.run_shard(
                        task_id, output_root=output, episode_root=root
                    )

            report = pilot.aggregate_shards(
                output_root=output, episode_root=root
            )
            self.assertEqual(report["artifact_verification"]["verified_shards"], 24)
            self.assertEqual(report["artifact_verification"]["verified_records"], 1_728)
            self.assertEqual(len(report["block_cell_summaries"]), 24)
            summary = report["block_cell_summaries"][0]
            self.assertEqual(summary["coverage"]["joint"]["successes"], 72)
            self.assertEqual(summary["coverage"]["joint"]["total"], 72)
            self.assertEqual(summary["interval_width"]["aggregate"]["median"], 0.2)
            self.assertFalse(summary["coverage"]["reference_centers_independent"])
            self.assertFalse(summary["coverage"]["coverage_indicators_independent"])
            self.assertFalse(summary["coverage"]["exact_binomial_acceptance_valid"])
            self.assertFalse(summary["coverage"]["scientific_acceptance_interpreted"])
            self.assertEqual(
                report["false_adequacy_interpretation"]["status"],
                "not_interpreted",
            )
            self.assertFalse(report["scientific_gate_applied"])
            self.assertFalse(report["coverage_threshold_applied"])
            self.assertNotIn("passed", report)
            with self.assertRaises(FileExistsError):
                pilot.aggregate_shards(output_root=output, episode_root=root)

    def test_aggregate_rejects_missing_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                pilot.aggregate_shards(output_root=output, episode_root=root)

    def test_shard_validation_rejects_terminal_and_control_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy()
            ), mock.patch.object(
                pilot,
                "run_generated_replicate",
                side_effect=self._generated_record,
            ):
                pilot.run_shard(0, output_root=output, episode_root=root)
            artifact = pilot._load_json(pilot.shard_path(output, 0))

            terminal_tamper = dict(artifact)
            terminal_tamper["records"] = [
                dict(record) for record in artifact["records"]
            ]
            terminal_tamper["records"][0]["terminal"] = (
                "single_population_adequate_before_sensitivity_gate"
            )
            with self.assertRaisesRegex(
                ValueError, "terminal_from_intervals_and_controls"
            ):
                pilot._validate_shard(terminal_tamper, 0)

            control_tamper = dict(artifact)
            control_tamper["records"] = [dict(record) for record in artifact["records"]]
            control_tamper["records"][0]["controls"] = dict(
                artifact["records"][0]["controls"]
            )
            control_tamper["records"][0]["controls"]["all_pass"] = False
            with self.assertRaisesRegex(ValueError, "record.controls.all_pass"):
                pilot._validate_shard(control_tamper, 0)

    def test_v2_uses_distinct_identity_root_and_1024_draws(self) -> None:
        observed_draws: list[int] = []

        def generated(*args: object, **kwargs: object) -> dict[str, object]:
            observed_draws.append(int(kwargs["integration_draws"]))
            return self._generated_record(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode12"
            output = pilot.V2_CONFIGURATION.output_root(root)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy()
            ), mock.patch.object(
                pilot, "run_generated_replicate", side_effect=generated
            ):
                for task_id in range(pilot.SHARD_COUNT):
                    pilot.run_shard(
                        task_id,
                        output_root=output,
                        episode_root=root,
                        configuration=pilot.V2_CONFIGURATION,
                    )
            first = pilot._load_json(pilot.shard_path(output, 0))
            pilot._validate_shard(first, 0, pilot.V2_CONFIGURATION)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                pilot._validate_shard(first, 0, pilot.V1_CONFIGURATION)
            report = pilot.aggregate_shards(
                output_root=output,
                episode_root=root,
                configuration=pilot.V2_CONFIGURATION,
            )

        self.assertEqual(set(observed_draws), {1_024})
        self.assertEqual(first["schema_version"], pilot.V2_CONFIGURATION.schema_version)
        self.assertEqual(first["pilot_id"], "interval_refinement_v2_1024")
        self.assertEqual(first["configuration"]["integration_draws"], 1_024)
        self.assertIn("interval_refinement_v2_1024__block_01", first["seed_namespace_cell_id"])
        self.assertEqual(report["pilot_id"], "interval_refinement_v2_1024")
        self.assertEqual(report["configuration"]["integration_draws"], 1_024)
        self.assertEqual(report["artifact_verification"]["verified_records"], 1_728)
        self.assertFalse(report["scientific_gate_applied"])
        self.assertNotIn("passed", report)

    def test_v3_uses_distinct_identity_root_and_2048_draws(self) -> None:
        observed_draws: list[int] = []

        def generated(*args: object, **kwargs: object) -> dict[str, object]:
            observed_draws.append(int(kwargs["integration_draws"]))
            return self._generated_record(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode12"
            output = pilot.V3_CONFIGURATION.output_root(root)
            with mock.patch.object(
                pilot.EpisodePolicy, "load", return_value=self._policy()
            ), mock.patch.object(
                pilot, "run_generated_replicate", side_effect=generated
            ):
                for task_id in range(pilot.SHARD_COUNT):
                    pilot.run_shard(
                        task_id,
                        output_root=output,
                        episode_root=root,
                        configuration=pilot.V3_CONFIGURATION,
                    )
            first = pilot._load_json(pilot.shard_path(output, 0))
            pilot._validate_shard(first, 0, pilot.V3_CONFIGURATION)
            for older in (pilot.V1_CONFIGURATION, pilot.V2_CONFIGURATION):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    pilot._validate_shard(first, 0, older)
            report = pilot.aggregate_shards(
                output_root=output,
                episode_root=root,
                configuration=pilot.V3_CONFIGURATION,
            )

        self.assertEqual(set(observed_draws), {2_048})
        self.assertEqual(first["schema_version"], pilot.V3_CONFIGURATION.schema_version)
        self.assertEqual(first["pilot_id"], "interval_refinement_v3_2048")
        self.assertEqual(first["configuration"]["integration_draws"], 2_048)
        self.assertIn("interval_refinement_v3_2048__block_01", first["seed_namespace_cell_id"])
        self.assertEqual(report["pilot_id"], "interval_refinement_v3_2048")
        self.assertEqual(report["configuration"]["integration_draws"], 2_048)
        self.assertEqual(report["artifact_verification"]["verified_records"], 1_728)
        self.assertFalse(report["scientific_gate_applied"])
        self.assertNotIn("passed", report)

    def test_module_cli_returns_zero_on_computational_completion(self) -> None:
        completed = {
            "task_id": 0,
            "records": [{}] * 72,
            "computationally_complete": True,
        }
        with mock.patch.object(pilot, "run_shard", return_value=completed), mock.patch(
            "builtins.print"
        ):
            self.assertEqual(pilot.main(["shard", "--task-id", "0"]), 0)
        aggregate = {
            "artifact_verification": {
                "verified_shards": 24,
                "verified_records": 1_728,
            }
        }
        with mock.patch.object(
            pilot, "aggregate_shards", return_value=aggregate
        ), mock.patch("builtins.print"):
            self.assertEqual(pilot.main(["aggregate"]), 0)

        with mock.patch.object(
            pilot, "run_shard", return_value=completed
        ) as run_v2, mock.patch("builtins.print"):
            self.assertEqual(
                pilot.main(
                    ["--configuration", "v2", "shard", "--task-id", "0"]
                ),
                0,
            )
        self.assertIs(
            run_v2.call_args.kwargs["configuration"], pilot.V2_CONFIGURATION
        )

        with mock.patch.object(
            pilot, "run_shard", return_value=completed
        ) as run_v3, mock.patch("builtins.print"):
            self.assertEqual(
                pilot.main(
                    ["--configuration", "v3", "shard", "--task-id", "0"]
                ),
                0,
            )
        self.assertIs(
            run_v3.call_args.kwargs["configuration"], pilot.V3_CONFIGURATION
        )

    def test_slurm_wrappers_are_serial_bounded_and_generated_only(self) -> None:
        array = (
            SLURM_ROOT / "pilot_interval_refinement_array.sbatch"
        ).read_text(encoding="utf-8")
        aggregate = (
            SLURM_ROOT / "pilot_interval_refinement_aggregate.sbatch"
        ).read_text(encoding="utf-8")
        array_v2 = (
            SLURM_ROOT / "pilot_interval_refinement_v2_1024_array.sbatch"
        ).read_text(encoding="utf-8")
        aggregate_v2 = (
            SLURM_ROOT / "pilot_interval_refinement_v2_1024_aggregate.sbatch"
        ).read_text(encoding="utf-8")
        array_v3 = (
            SLURM_ROOT / "pilot_interval_refinement_v3_2048_array.sbatch"
        ).read_text(encoding="utf-8")
        aggregate_v3 = (
            SLURM_ROOT / "pilot_interval_refinement_v3_2048_aggregate.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-23", array)
        self.assertIn("#SBATCH --array=0-23", array_v2)
        self.assertIn("#SBATCH --array=0-23", array_v3)
        for text in (array, aggregate, array_v2, aggregate_v2, array_v3, aggregate_v3):
            self.assertIn("#SBATCH --ntasks=1", text)
            self.assertIn("#SBATCH --cpus-per-task=1", text)
            self.assertIn("#SBATCH --mem=6G", text)
            self.assertIn("ep12_executor.interval_refinement_pilot", text)
            self.assertIn("/scratch/users/zijiao/", text)
            self.assertNotIn("final_evaluation", text)
            self.assertNotIn("materialize-development", text)
            self.assertLess(
                text.index(". /etc/profile >/dev/null"),
                text.index("set -euo pipefail"),
            )
            self.assertLess(
                text.index("set -euo pipefail"), text.index("module purge")
            )
        self.assertIn("#SBATCH --time=02:00:00", array)
        self.assertIn("#SBATCH --time=02:00:00", array_v2)
        self.assertIn("#SBATCH --time=02:00:00", array_v3)
        self.assertIn("#SBATCH --time=00:15:00", aggregate)
        self.assertIn("#SBATCH --time=00:15:00", aggregate_v2)
        self.assertIn("#SBATCH --time=00:15:00", aggregate_v3)
        for text in (array_v2, aggregate_v2):
            self.assertIn("--configuration v2", text)
            self.assertIn("interval-refinement-v2-1024", text)
        for text in (array_v3, aggregate_v3):
            self.assertIn("--configuration v3", text)
            self.assertIn("interval-refinement-v3-2048", text)


if __name__ == "__main__":
    unittest.main()
