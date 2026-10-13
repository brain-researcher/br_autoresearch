from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np

    from ep12_executor import integration_refinement as refinement

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class IntegrationRefinementTests(unittest.TestCase):
    @staticmethod
    def _fixture(scenario: str, **settings: object) -> SimpleNamespace:
        seed = int(settings["seed"])
        left = SimpleNamespace(
            counts=("left", scenario, seed),
            covariates=("left-covariates", scenario, seed),
        )
        right = SimpleNamespace(
            counts=("right", scenario, seed),
            covariates=("right-covariates", scenario, seed),
        )
        return SimpleNamespace(left=left, right=right)

    @staticmethod
    def _transfer(
        left_counts: tuple[object, ...],
        left_covariates: tuple[object, ...],
        right_counts: tuple[object, ...],
        right_covariates: tuple[object, ...],
        **settings: object,
    ) -> dict[str, object]:
        del left_covariates, right_counts, right_covariates
        draws = int(settings["integration_draws"])
        scenario = str(left_counts[1])
        seed = int(left_counts[2])
        base = refinement.SCENARIOS.index(scenario) * 0.01 + seed * 1e-7
        offsets = {256: 0.04, 512: 0.0015, 1024: 0.0, 2048: -0.0007}
        metadata_tag = "old" if draws == 256 else "candidate"
        metadata = {
            "left": {
                "U": f"U-{metadata_tag}",
                "M": "M2",
                "reference": "T",
                "null_generator": "T",
            },
            "right": {
                "U": f"U-{metadata_tag}",
                "M": "M2",
                "reference": "T",
                "null_generator": "T",
            },
        }
        return {
            "left_to_right": {
                "mean_delta_M_minus_reference": base + offsets[draws]
            },
            "right_to_left": {
                "mean_delta_M_minus_reference": -base - offsets[draws]
            },
            "source_fit_metadata": metadata,
        }

    def test_actual_sobol_generator_has_exact_nested_prefixes(self) -> None:
        result = refinement.validate_sobol_prefixes()
        self.assertTrue(result["passed"])
        self.assertEqual(result["grid"], [256, 512, 1024])
        self.assertTrue(
            all(check["exact_prefix_match"] for check in result["checks"])
        )

    def test_report_compares_512_to_1024_and_replays_exactly(self) -> None:
        with mock.patch.object(
            refinement, "generate_synthetic_type", side_effect=self._fixture
        ), mock.patch.object(
            refinement, "reciprocal_transfer", side_effect=self._transfer
        ):
            report = refinement.build_refinement_report()

        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate_draws"], 512)
        self.assertEqual(report["high_draws"], 1024)
        self.assertEqual(report["absolute_delta_change_tolerance"], 0.002)
        self.assertAlmostEqual(
            report["maximum_observed_absolute_delta_change"], 0.0015
        )
        self.assertTrue(report["all_selected_model_metadata_exactly_match"])
        self.assertTrue(report["deterministic_replay"]["passed"])
        self.assertFalse(report["real_connectivity_accessed"])
        self.assertEqual(len(report["rows"]), 6)

    def test_successor_compares_1024_to_2048_with_unchanged_rules(self) -> None:
        with mock.patch.object(
            refinement, "generate_synthetic_type", side_effect=self._fixture
        ), mock.patch.object(
            refinement, "reciprocal_transfer", side_effect=self._transfer
        ):
            report = refinement.build_refinement_report(
                draw_grid=(512, 1024, 2048),
                candidate_draws=1024,
                high_draws=2048,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["integration_draws_grid"], [512, 1024, 2048])
        self.assertEqual(report["candidate_draws"], 1024)
        self.assertEqual(report["high_draws"], 2048)
        self.assertEqual(report["absolute_delta_change_tolerance"], 0.002)
        self.assertEqual(
            report["selected_model_criterion"],
            "exact_source_fit_metadata_equality",
        )
        self.assertAlmostEqual(
            report["maximum_observed_absolute_delta_change"], 0.0007
        )
        self.assertTrue(report["deterministic_replay"]["passed"])
        self.assertEqual(len(report["rows"]), 6)

    def test_invalid_successor_draw_configuration_fails_closed(self) -> None:
        for settings in (
            ((512, 1024, 1024), 1024, 1024),
            ((1024, 512, 2048), 1024, 2048),
            ((512, 1024, 2048), 2048, 1024),
            ((512, 1024, 2048), 768, 2048),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                refinement.build_refinement_report(
                    draw_grid=settings[0],
                    candidate_draws=settings[1],
                    high_draws=settings[2],
                )

    def test_exact_candidate_high_metadata_mismatch_fails(self) -> None:
        def mismatching_transfer(*args: object, **kwargs: object) -> dict[str, object]:
            result = self._transfer(*args, **kwargs)
            if int(kwargs["integration_draws"]) == 1024:
                result["source_fit_metadata"]["left"]["reference"] = "U-curved"
            return result

        with mock.patch.object(
            refinement, "generate_synthetic_type", side_effect=self._fixture
        ), mock.patch.object(
            refinement, "reciprocal_transfer", side_effect=mismatching_transfer
        ):
            report = refinement.build_refinement_report()

        self.assertFalse(report["passed"])
        self.assertFalse(report["all_selected_model_metadata_exactly_match"])
        self.assertFalse(report["numerical_criterion_passed"])

    def test_deterministic_replay_mismatch_fails_closed(self) -> None:
        calls = 0

        def drifting_transfer(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            result = self._transfer(*args, **kwargs)
            if calls > 18:
                result["left_to_right"]["mean_delta_M_minus_reference"] += 1e-8
            return result

        with mock.patch.object(
            refinement, "generate_synthetic_type", side_effect=self._fixture
        ), mock.patch.object(
            refinement, "reciprocal_transfer", side_effect=drifting_transfer
        ):
            report = refinement.build_refinement_report()

        self.assertFalse(report["passed"])
        self.assertFalse(report["deterministic_replay"]["passed"])

    def test_atomic_publish_rejects_overwrite_and_outside_path(self) -> None:
        canned = {
            "passed": True,
            "real_connectivity_accessed": False,
            "maximum_observed_absolute_delta_change": 0.001,
            "all_selected_model_metadata_exactly_match": True,
            "deterministic_replay": {"passed": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode12"
            output = (
                root
                / "outputs"
                / "scientific_qualification"
                / "pilots"
                / "unit-test-run"
            )
            with mock.patch.object(
                refinement, "build_refinement_report", return_value=canned
            ):
                result = refinement.run_integration_refinement(
                    output, episode_root=root
                )
                self.assertEqual(result, canned)
                written = json.loads(
                    (output / refinement.REPORT_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(written, canned)
                with self.assertRaises(FileExistsError):
                    refinement.run_integration_refinement(output, episode_root=root)
                with self.assertRaises(ValueError):
                    refinement.run_integration_refinement(
                        root / "outputs" / "outside", episode_root=root
                    )

    def test_cli_returns_failure_for_a_published_failed_gate(self) -> None:
        failed = {
            "passed": False,
            "maximum_observed_absolute_delta_change": 0.0021,
            "all_selected_model_metadata_exactly_match": True,
            "deterministic_replay": {"passed": True},
        }
        with mock.patch.object(
            refinement, "run_integration_refinement", return_value=failed
        ), mock.patch("builtins.print"):
            status = refinement.main(
                [
                    "--output-dir",
                    str(refinement.DEFAULT_OUTPUT_DIR.with_name("unit-cli")),
                ]
            )
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
