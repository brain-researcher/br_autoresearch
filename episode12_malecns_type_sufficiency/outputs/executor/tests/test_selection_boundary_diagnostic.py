from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np

    from ep12_executor import selection_boundary_diagnostic as diagnostic
    from ep12_executor.scientific_models import CandidateSelection

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class SelectionBoundaryDiagnosticTests(unittest.TestCase):
    def test_existing_near_set_rule_and_signed_boundary_distance(self) -> None:
        summaries = [
            CandidateSelection("M2", 0.91, 0.02, (0.89, 0.93), 20),
            CandidateSelection("M3", 1.00, 0.10, (0.90, 1.10), 30),
        ]
        result = diagnostic._candidate_selection_diagnostics(summaries)

        self.assertEqual(result["best_score_model_id"], "M3")
        self.assertEqual(result["near_set_model_ids"], ["M2", "M3"])
        self.assertEqual(result["selected_model_id"], "M2")
        by_model = {row["model_id"]: row for row in result["candidates"]}
        self.assertAlmostEqual(
            by_model["M2"]["near_set_boundary_signed_distance"], 0.01
        )
        self.assertTrue(by_model["M2"]["in_near_set"])

    def test_candidate_cv_shares_fold_integration_seeds(self) -> None:
        counts = np.ones((6, 3), dtype=np.int64)
        covariates = np.arange(6, dtype=float)[:, None]
        calls: list[tuple[str, int, int]] = []

        class FakeModel:
            def __init__(self, model_id: str) -> None:
                self.model_id = model_id
                self.parameter_count = 20 if model_id == "M2" else 30

            def mean_score(self, target_counts: object, target_covariates: object) -> float:
                del target_counts, target_covariates
                return 1.0 if self.model_id == "M2" else 0.9

        def fake_fit(
            source_counts: object,
            source_covariates: object,
            model_id: str,
            **settings: object,
        ) -> FakeModel:
            del source_counts, source_covariates
            calls.append(
                (
                    model_id,
                    int(settings["integration_seed"]),
                    int(settings["integration_draws"]),
                )
            )
            return FakeModel(model_id)

        with mock.patch.object(
            diagnostic, "fit_predictive_model", side_effect=fake_fit
        ):
            result = diagnostic._evaluate_m_candidates(
                counts,
                covariates,
                source_seed=123,
                integration_draws=512,
            )

        self.assertEqual(result["selected_model_id"], "M2")
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][1], calls[2][1])
        self.assertEqual(calls[1][1], calls[3][1])
        self.assertTrue(
            result["integration_seed_shared_between_M2_and_M3_within_fold"]
        )

    def test_draw_scores_both_fixed_M_models_against_one_reference(self) -> None:
        fixture = SimpleNamespace(
            left=SimpleNamespace(
                counts=np.ones((4, 3), dtype=np.int64),
                covariates=np.zeros((4, 1), dtype=float),
            ),
            right=SimpleNamespace(
                counts=np.ones((2, 3), dtype=np.int64),
                covariates=np.zeros((2, 1), dtype=float),
            ),
        )

        class Scorer:
            def __init__(self, model_id: str, scores: list[float]) -> None:
                self.model_id = model_id
                self._scores = np.asarray(scores, dtype=float)

            def score_neurons(self, counts: object, covariates: object) -> np.ndarray:
                del counts, covariates
                return self._scores.copy()

        triplet = SimpleNamespace(
            T=Scorer("T", [0.0, 0.0]),
            U=Scorer("U_linear", [1.0, 1.0]),
            M_cv=SimpleNamespace(model_id="M2"),
            reference_model_id="U_linear",
        )
        selection = {
            "selected_model_id": "M2",
            "candidates": [],
        }
        fixed_calls: list[tuple[str, int]] = []

        def fake_fixed_fit(
            counts: object,
            covariates: object,
            model_id: str,
            **settings: object,
        ) -> Scorer:
            del counts, covariates
            fixed_calls.append((model_id, int(settings["integration_seed"])))
            scores = [1.1, 1.3] if model_id == "M2" else [0.9, 1.3]
            return Scorer(model_id, scores)

        with mock.patch.object(
            diagnostic, "fit_triplet", return_value=triplet
        ), mock.patch.object(
            diagnostic,
            "_evaluate_m_candidates",
            return_value=selection,
        ), mock.patch.object(
            diagnostic, "fit_predictive_model", side_effect=fake_fixed_fit
        ):
            result = diagnostic._evaluate_draw(
                fixture, source_seed=99, integration_draws=512
            )

        self.assertEqual(result["selected_reference_model_id"], "U_linear")
        deltas = result[
            "fixed_M_left_to_right_mean_delta_against_selected_reference"
        ]
        self.assertAlmostEqual(deltas["M2"], 0.2)
        self.assertAlmostEqual(deltas["M3"], 0.1)
        self.assertEqual(fixed_calls[0][1], fixed_calls[1][1])
        self.assertTrue(
            result["M_candidate_cross_validation"][
                "production_selection_exactly_reproduced"
            ]
        )

    def test_right_source_uses_right_fit_seed_and_left_target(self) -> None:
        fixture = SimpleNamespace(
            left=SimpleNamespace(counts="left-counts", covariates="left-covariates"),
            right=SimpleNamespace(
                counts="right-counts", covariates="right-covariates"
            ),
        )
        observed: list[tuple[int, int, str]] = []

        def evaluate(
            supplied_fixture: object,
            *,
            source_seed: int,
            integration_draws: int,
            source_side: str,
        ) -> dict[str, object]:
            self.assertIs(supplied_fixture, fixture)
            observed.append((source_seed, integration_draws, source_side))
            return {
                "integration_draws": integration_draws,
                "selected_reference_model_id": "U_linear",
            }

        with mock.patch.object(
            diagnostic, "generate_synthetic_type", return_value=fixture
        ) as generated, mock.patch.object(
            diagnostic, "_evaluate_draw", side_effect=evaluate
        ):
            rows, _ = diagnostic._evaluate_once(
                scenario="curved_continuum_dense",
                fixture_seed=1201,
                source_side="right",
                draw_grid=(1024, 2048, 4096),
            )

        transfer_seed = diagnostic.seed_from(
            "integration-stability", "curved_continuum_dense", 1201
        )
        expected_source_seed = diagnostic.seed_from(transfer_seed, "right-source")
        self.assertEqual(
            observed,
            [
                (expected_source_seed, 1024, "right"),
                (expected_source_seed, 2048, "right"),
                (expected_source_seed, 4096, "right"),
            ],
        )
        self.assertEqual([row["integration_draws"] for row in rows], [1024, 2048, 4096])
        generated.assert_called_once_with(
            "curved_continuum_dense",
            left_neurons=diagnostic.NEURONS_PER_SIDE,
            right_neurons=diagnostic.NEURONS_PER_SIDE,
            count_total=diagnostic.COUNT_TOTAL,
            strength=diagnostic.GENERATOR_STRENGTH,
            missing_fraction=diagnostic.MISSING_FRACTION,
            seed=1201,
        )

    def test_report_exact_replay_is_diagnostic_not_gate(self) -> None:
        rows = [
            {
                "integration_draws": draws,
                "selected_reference_model_id": "U_linear",
            }
            for draws in diagnostic.DRAW_GRID
        ]
        with mock.patch.object(
            diagnostic,
            "_evaluate_once",
            side_effect=[(rows, {"wall_seconds": 1.0, "cpu_seconds": 0.9})] * 2,
        ):
            report = diagnostic.build_selection_boundary_report()

        self.assertTrue(report["deterministic_replay"]["exact_row_replay"])
        self.assertFalse(report["scientific_gate_applied"])
        self.assertFalse(report["acceptance_threshold_applied"])
        self.assertNotIn("passed", report)
        self.assertEqual(report["integration_draws_grid"], [512, 1024, 2048])

    def test_custom_report_records_right_direction_without_a_threshold(self) -> None:
        rows = [
            {
                "integration_draws": draws,
                "selected_reference_model_id": "U_linear",
            }
            for draws in (1024, 2048, 4096)
        ]
        timing = {"wall_seconds": 1.0, "cpu_seconds": 0.9}
        with mock.patch.object(
            diagnostic, "_evaluate_once", side_effect=[(rows, timing), (rows, timing)]
        ) as evaluate:
            report = diagnostic.build_selection_boundary_report(
                scenario="curved_continuum_dense",
                fixture_seed=1201,
                source_side="right",
                draw_grid=(1024, 2048, 4096),
            )

        self.assertEqual(report["fixture_configuration"]["fixture_seed"], 1201)
        self.assertEqual(
            report["fixture_configuration"]["direction"], "right_to_left"
        )
        self.assertEqual(report["integration_draws_grid"], [1024, 2048, 4096])
        self.assertTrue(report["deterministic_replay"]["exact_row_replay"])
        self.assertFalse(report["acceptance_threshold_applied"])
        self.assertNotIn("passed", report)
        expected_kwargs = {
            "scenario": "curved_continuum_dense",
            "fixture_seed": 1201,
            "source_side": "right",
            "draw_grid": (1024, 2048, 4096),
        }
        self.assertEqual(evaluate.call_args_list[0].kwargs, expected_kwargs)
        self.assertEqual(evaluate.call_args_list[1].kwargs, expected_kwargs)

    def test_configuration_is_closed_and_draw_grid_is_bounded(self) -> None:
        invalid = (
            ("not_a_scenario", 1201, "right", (1024, 2048)),
            ("curved_continuum_dense", -1, "right", (1024, 2048)),
            ("curved_continuum_dense", 1201, "bilateral", (1024, 2048)),
            ("curved_continuum_dense", 1201, "right", (2048, 1024)),
            ("curved_continuum_dense", 1201, "right", (1024, 1024)),
            ("curved_continuum_dense", 1201, "right", (1024,)),
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                diagnostic.build_selection_boundary_report(
                    scenario=settings[0],
                    fixture_seed=settings[1],
                    source_side=settings[2],
                    draw_grid=settings[3],
                )

    def test_atomic_write_once_and_output_boundary(self) -> None:
        canned = {
            "schema_version": diagnostic.SCHEMA_VERSION,
            "integration_draws_grid": [512, 1024, 2048],
            "scientific_gate_applied": False,
            "deterministic_replay": {"exact_row_replay": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode12"
            output = (
                root
                / "outputs"
                / "scientific_qualification"
                / "pilots"
                / "unit-selection-boundary"
            )
            with mock.patch.object(
                diagnostic,
                "build_selection_boundary_report",
                return_value=canned,
            ):
                result = diagnostic.run_selection_boundary_diagnostic(
                    output, episode_root=root
                )
                self.assertEqual(result, canned)
                written = json.loads(
                    (output / diagnostic.REPORT_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(written, canned)
                with self.assertRaises(FileExistsError):
                    diagnostic.run_selection_boundary_diagnostic(
                        output, episode_root=root
                    )
                with self.assertRaises(ValueError):
                    diagnostic.run_selection_boundary_diagnostic(
                        root / "outputs" / "outside", episode_root=root
                    )

    def test_cli_fails_only_for_nonreproducible_rows(self) -> None:
        report = {
            "integration_draws_grid": [512, 1024, 2048],
            "scientific_gate_applied": False,
            "deterministic_replay": {"exact_row_replay": False},
        }
        with mock.patch.object(
            diagnostic,
            "run_selection_boundary_diagnostic",
            return_value=report,
        ), mock.patch("builtins.print"):
            status = diagnostic.main(
                ["--output-dir", str(diagnostic.DEFAULT_OUTPUT_DIR)]
            )
        self.assertEqual(status, 1)

    def test_cli_forwards_focused_right_source_configuration(self) -> None:
        report = {
            "integration_draws_grid": [1024, 2048, 4096],
            "scientific_gate_applied": False,
            "deterministic_replay": {"exact_row_replay": True},
        }
        output = diagnostic.DEFAULT_OUTPUT_DIR.with_name("unit-right-source")
        with mock.patch.object(
            diagnostic,
            "run_selection_boundary_diagnostic",
            return_value=report,
        ) as run, mock.patch("builtins.print"):
            status = diagnostic.main(
                [
                    "--output-dir",
                    str(output),
                    "--scenario",
                    "curved_continuum_dense",
                    "--fixture-seed",
                    "1201",
                    "--source-side",
                    "right",
                    "--draw-grid",
                    "1024",
                    "2048",
                    "4096",
                ]
            )
        self.assertEqual(status, 0)
        run.assert_called_once_with(
            output,
            scenario="curved_continuum_dense",
            fixture_seed=1201,
            source_side="right",
            draw_grid=[1024, 2048, 4096],
        )

    def test_focused_right_source_wrapper_is_generated_only_and_bounded(self) -> None:
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "slurm"
            / "pilot_selection_boundary_curved1201_right_1024_2048_4096.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --ntasks=1", wrapper)
        self.assertIn("#SBATCH --cpus-per-task=1", wrapper)
        self.assertIn("#SBATCH --mem=6G", wrapper)
        self.assertIn("#SBATCH --time=00:30:00", wrapper)
        self.assertIn("--fixture-seed 1201", wrapper)
        self.assertIn("--source-side right", wrapper)
        self.assertIn("--draw-grid 1024 2048 4096", wrapper)
        self.assertIn("ep12_executor.selection_boundary_diagnostic", wrapper)
        self.assertNotIn("final_evaluation", wrapper)


if __name__ == "__main__":
    unittest.main()
