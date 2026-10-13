from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from ep12_executor import interval_refinement_pilot as pilot
    from ep12_executor import measured_gain_calibration as calibration

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class MeasuredGainCalibrationTests(unittest.TestCase):
    @staticmethod
    def _root(
        temporary: str,
        configuration: object = None,
    ) -> tuple[Path, Path]:
        root = Path(temporary) / "episode12"
        selected = configuration or pilot.V1_CONFIGURATION
        output = selected.output_root(root)
        return root, output

    @staticmethod
    def _policy() -> SimpleNamespace:
        return SimpleNamespace(meaningful_margin=0.02, policy_hash="a" * 64)

    @staticmethod
    def _block_one_deltas(scenario: str) -> list[float]:
        if scenario == "residual_groups_within_continuum":
            return [0.031, 0.032]
        if scenario == "weak_residual_groups":
            return [0.010, 0.011]
        if scenario == "linear_continuum":
            return [0.031, 0.011]
        return [0.0, 0.0]

    @classmethod
    def _generated_record(
        cls,
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
        assert integration_draws in (512, 1_024, 2_048)
        assert folds == 2
        block_one = "block_01" in cell.cell_id
        if block_one:
            deltas = cls._block_one_deltas(cell.scenario)
        else:
            # Deliberately incompatible with every block-01 label.  The
            # calibration must remain unchanged because block 02 is terminal-only.
            deltas = [99.0, -99.0]
        adequate = (
            not block_one
            and cell.scenario == "residual_groups_within_continuum"
            and replicate_index < 3
        )
        weak_adequate = (
            not block_one
            and cell.scenario == "weak_residual_groups"
            and replicate_index < 9
        )
        terminal = (
            "single_population_adequate_before_sensitivity_gate"
            if adequate or weak_adequate
            else "unresolved"
        )
        if adequate or weak_adequate:
            interval_lower = [-0.01, -0.01, -0.01]
            interval_upper = [0.01, 0.01, 0.01]
        else:
            interval_lower = [-0.1, -0.1, -0.1]
            interval_upper = [0.1, 0.1, 0.1]
        return {
            "cell_id": cell.cell_id,
            "scenario": cell.scenario,
            "truth_family": pilot.SCENARIO_TRUTH[cell.scenario],
            "replicate_index": replicate_index,
            "terminal": terminal,
            "intervals": {
                "method": "paired_whole_type_studentized_bootstrap_max_t",
                "confidence_level": 0.95,
                "labels": ["left_to_right", "right_to_left", "aggregate"],
                "estimate": [0.0, 0.0, 0.0],
                "standard_error": [0.01, 0.01, 0.005],
                "critical_max_t": 2.0,
                "lower": interval_lower,
                "upper": interval_upper,
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
                    list(deltas) for _ in range(whole_types)
                ],
                "influence_control_pass": True,
                "all_pass": True,
            },
            "numerical_failures": [],
            "type_records": [
                {
                    "synthetic_type_index": index,
                    "deltas": list(deltas),
                    "source_fit_metadata": {"block_one": block_one},
                }
                for index in range(whole_types)
            ],
            "wall_seconds": 0.1,
            "cpu_seconds": 0.1,
        }

    @classmethod
    def _write_shards(
        cls,
        root: Path,
        output: Path,
        *,
        configuration: object = None,
    ) -> None:
        selected = configuration or pilot.V1_CONFIGURATION
        with mock.patch.object(
            pilot.EpisodePolicy, "load", return_value=cls._policy()
        ), mock.patch.object(
            pilot, "run_generated_replicate", side_effect=cls._generated_record
        ):
            for task_id in range(pilot.SHARD_COUNT):
                pilot.run_shard(
                    task_id,
                    output_root=output,
                    episode_root=root,
                    configuration=selected,
                )

    @staticmethod
    def _result(report: dict[str, object], scenario: str) -> dict[str, object]:
        return next(
            item
            for item in report["cell_results"]
            if item["cell"]["scenario"] == scenario
            and item["cell"]["neurons_per_side"] == 7
        )

    def test_split_calibration_labels_and_false_adequacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            report = calibration.publish_measured_gain_calibration(
                output_root=output, episode_root=root
            )

        above = self._result(report, "residual_groups_within_continuum")
        below = self._result(report, "weak_residual_groups")
        indeterminate = self._result(report, "linear_continuum")
        self.assertEqual(
            above["calibration"]["margin_classification"], "above_margin"
        )
        self.assertEqual(
            below["calibration"]["margin_classification"], "below_margin"
        )
        self.assertEqual(
            indeterminate["calibration"]["margin_classification"],
            "margin_indeterminate",
        )
        self.assertEqual(above["evaluation"]["adequacy_terminal_count"], 3)
        self.assertEqual(above["false_adequacy"]["successes"], 3)
        self.assertEqual(above["false_adequacy"]["total"], 72)
        self.assertEqual(above["false_adequacy"]["rate"], 3 / 72)
        self.assertEqual(
            above["false_adequacy"]["clopper_pearson_exact_95_interval"],
            list(pilot.exact_binomial_interval(3, 72)),
        )
        self.assertEqual(below["evaluation"]["adequacy_terminal_count"], 9)
        self.assertEqual(
            below["false_adequacy"]["status"], "unresolved_for_adequacy"
        )
        self.assertIsNone(below["false_adequacy"]["rate"])
        self.assertEqual(
            indeterminate["false_adequacy"]["status"],
            "not_applicable_non_residual_group_generator",
        )
        self.assertIsNone(indeterminate["false_adequacy"]["total"])
        self.assertFalse(report["population_oracle_truth"])
        self.assertFalse(report["scientific_gate_applied"])
        self.assertFalse(report["scientific_pass_fail_assigned"])
        self.assertFalse(report["real_connectivity_accessed"])
        self.assertNotIn("passed", report)

    def test_blocks_are_disjoint_output_has_no_fitted_record_leak_and_is_write_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            report = calibration.publish_measured_gain_calibration(
                output_root=output, episode_root=root
            )
            with self.assertRaises(FileExistsError):
                calibration.publish_measured_gain_calibration(
                    output_root=output, episode_root=root
                )
            encoded = json.dumps(report, sort_keys=True)

        verification = report["artifact_verification"]
        self.assertTrue(verification["block_seed_namespaces_disjoint"])
        self.assertTrue(verification["block_record_identities_disjoint"])
        self.assertFalse(verification["evaluation_fitted_delta_records_used"])
        calibration_namespaces = set(
            report["block_roles"]["calibration"]["seed_namespaces"]
        )
        evaluation_namespaces = set(
            report["block_roles"]["evaluation"]["seed_namespaces"]
        )
        self.assertTrue(calibration_namespaces.isdisjoint(evaluation_namespaces))
        self.assertEqual(
            report["block_roles"]["calibration"]["fitted_delta_rows"], 6_912
        )
        self.assertNotIn('"deltas"', encoded)
        self.assertNotIn('"type_records"', encoded)
        self.assertNotIn('"source_fit_metadata"', encoded)

    def test_tampered_calibration_delta_fails_closed_without_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            path = pilot.shard_path(output, 0)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["records"][0]["type_records"][0]["deltas"] = [0.03]
            path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "two finite directions"):
                calibration.publish_measured_gain_calibration(
                    output_root=output, episode_root=root
                )
            self.assertFalse((output / calibration.ARTIFACT_NAME).exists())

    def test_missing_shard_and_namespace_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            pilot.shard_path(output, 23).unlink()
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                calibration.publish_measured_gain_calibration(
                    output_root=output, episode_root=root
                )

        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            path = pilot.shard_path(output, 12)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["seed_namespace_cell_id"] = artifact[
                "seed_namespace_cell_id"
            ].replace("block_02", "block_01")
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed_namespace_cell_id"):
                calibration.publish_measured_gain_calibration(
                    output_root=output, episode_root=root
                )
            self.assertFalse((output / calibration.ARTIFACT_NAME).exists())

    def test_block_two_terminal_tamper_fails_before_false_adequacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary)
            self._write_shards(root, output)
            path = pilot.shard_path(output, 12)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["records"][0]["terminal"] = (
                "single_population_adequate_before_sensitivity_gate"
            )
            path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "terminal_from_intervals_and_controls"
            ):
                calibration.publish_measured_gain_calibration(
                    output_root=output, episode_root=root
                )
            self.assertFalse((output / calibration.ARTIFACT_NAME).exists())

    def test_v2_uses_distinct_root_schema_and_seed_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, v1_output = self._root(temporary)
            _, v2_output = self._root(temporary, pilot.V2_CONFIGURATION)
            self._write_shards(root, v1_output)
            self._write_shards(
                root,
                v2_output,
                configuration=pilot.V2_CONFIGURATION,
            )
            v1_report = calibration.publish_measured_gain_calibration(
                output_root=v1_output,
                episode_root=root,
            )
            v2_report = calibration.publish_measured_gain_calibration(
                output_root=v2_output,
                episode_root=root,
                configuration=pilot.V2_CONFIGURATION,
            )
            v1_artifact_exists = (
                v1_output / calibration.ARTIFACT_NAME
            ).is_file()
            v2_artifact_exists = (
                v2_output / calibration.ARTIFACT_NAME
            ).is_file()

        self.assertEqual(v1_report["schema_version"], calibration.SCHEMA_VERSION)
        self.assertEqual(v1_report["pilot_id"], pilot.PILOT_ID)
        self.assertEqual(
            v2_report["schema_version"], calibration.V2_SCHEMA_VERSION
        )
        self.assertEqual(v2_report["pilot_id"], pilot.V2_CONFIGURATION.pilot_id)
        v1_cell = v1_report["cell_results"][0]["calibration"]
        v2_cell = v2_report["cell_results"][0]["calibration"]
        self.assertEqual(
            v1_cell["bootstrap_seed_namespace"],
            calibration.CALIBRATION_SEED_NAMESPACE,
        )
        self.assertEqual(
            v2_cell["bootstrap_seed_namespace"],
            calibration.V2_CALIBRATION_SEED_NAMESPACE,
        )
        self.assertNotEqual(v1_cell["bootstrap_seed"], v2_cell["bootstrap_seed"])
        self.assertTrue(v1_artifact_exists)
        self.assertTrue(v2_artifact_exists)
        self.assertFalse(v2_report["scientific_gate_applied"])
        self.assertFalse(v2_report["scientific_pass_fail_assigned"])

    def test_v2_configuration_rejects_v1_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, v1_output = self._root(temporary)
            with self.assertRaisesRegex(
                ValueError, "interval refinement artifacts must remain"
            ):
                calibration.publish_measured_gain_calibration(
                    output_root=v1_output,
                    episode_root=root,
                    configuration=pilot.V2_CONFIGURATION,
                )

    def test_v3_uses_distinct_root_schema_and_seed_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, output = self._root(temporary, pilot.V3_CONFIGURATION)
            self._write_shards(
                root,
                output,
                configuration=pilot.V3_CONFIGURATION,
            )
            report = calibration.publish_measured_gain_calibration(
                output_root=output,
                episode_root=root,
                configuration=pilot.V3_CONFIGURATION,
            )
            artifact_exists = (output / calibration.ARTIFACT_NAME).is_file()

        self.assertEqual(report["schema_version"], calibration.V3_SCHEMA_VERSION)
        self.assertEqual(report["pilot_id"], pilot.V3_CONFIGURATION.pilot_id)
        cell = report["cell_results"][0]["calibration"]
        self.assertEqual(
            cell["bootstrap_seed_namespace"],
            calibration.V3_CALIBRATION_SEED_NAMESPACE,
        )
        self.assertNotEqual(
            cell["bootstrap_seed_namespace"],
            calibration.V2_CALIBRATION_SEED_NAMESPACE,
        )
        self.assertTrue(artifact_exists)
        self.assertFalse(report["scientific_gate_applied"])
        self.assertFalse(report["scientific_pass_fail_assigned"])

    def test_main_selects_v2_and_no_argument_remains_v1(self) -> None:
        report = {
            "artifact_verification": {"verified_shards": pilot.SHARD_COUNT},
            "cell_results": [{} for _ in pilot.PILOT_CELLS],
        }
        with mock.patch.object(
            calibration,
            "publish_measured_gain_calibration",
            return_value=report,
        ) as publish, mock.patch("builtins.print"):
            self.assertEqual(calibration.main([]), 0)
            publish.assert_called_once_with(
                output_root=pilot.V1_CONFIGURATION.output_root(),
                configuration=pilot.V1_CONFIGURATION,
            )

        with mock.patch.object(
            calibration,
            "publish_measured_gain_calibration",
            return_value=report,
        ) as publish, mock.patch("builtins.print"):
            self.assertEqual(calibration.main(["--configuration", "v2"]), 0)
            publish.assert_called_once_with(
                output_root=pilot.V2_CONFIGURATION.output_root(),
                configuration=pilot.V2_CONFIGURATION,
            )

        with mock.patch.object(
            calibration,
            "publish_measured_gain_calibration",
            return_value=report,
        ) as publish, mock.patch("builtins.print"):
            self.assertEqual(calibration.main(["--configuration", "v3"]), 0)
            publish.assert_called_once_with(
                output_root=pilot.V3_CONFIGURATION.output_root(),
                configuration=pilot.V3_CONFIGURATION,
            )


if __name__ == "__main__":
    unittest.main()
