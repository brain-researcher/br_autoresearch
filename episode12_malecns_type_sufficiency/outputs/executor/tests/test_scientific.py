from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from ep12_executor.runtime import EPISODE_ROOT
from ep12_executor.yaml_subset import load_yaml_subset

try:
    import numpy as np
    from scipy import sparse

    from ep12_executor.development_trial import _collapse_vocabulary, _endpoint_bin
    from ep12_executor.scientific_models import (
        BOUNDED_LATENT_SUPPORT_PROTOCOL,
        CandidateSelection,
        _cubic_hermite_basis,
        cross_validated_candidate,
        fit_predictive_model,
        fit_triplet,
        helmert_ilr_basis,
        seed_from,
        validate_profile_arrays,
    )
    from ep12_executor.scientific_qualification import (
        QualificationCell,
        _accept_cell,
        _cell_summary,
        _contract_settings,
        _require_active_model_family_conformance,
        exact_binomial_interval,
        paired_simultaneous_intervals,
    )
    from ep12_executor.scientific_synthetic import PARTNER_NAMES, generate_synthetic_type

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


class ScientificContractTests(unittest.TestCase):
    def test_contract_keeps_real_connectivity_closed_and_bounds_u(self) -> None:
        contract = load_yaml_subset(
            EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
        )
        self.assertEqual(contract["real_connectivity_access_authorized"], "development_only")
        self.assertFalse(contract["final_connectivity_access_authorized"])
        self.assertEqual(
            contract["models"]["U_reference"]["candidates"],
            ["U_linear", "U_curved"],
        )
        self.assertEqual(contract["models"]["U_linear"]["latent_dimension_max"], 2)
        self.assertEqual(contract["models"]["U_curved"]["polynomial_degree"], 3)
        self.assertEqual(contract["models"]["M"]["component_counts"], [2, 3])
        self.assertEqual(
            contract["execution_parameters"][
                "predictive_integration_draws_per_proposal"
            ],
            256,
        )
        self.assertEqual(
            contract["execution_parameters"]["numeric_threads_per_worker"], 1
        )

    @unittest.skipUnless(
        SCIENTIFIC_STACK_AVAILABLE,
        "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
    )
    def test_v3_contract_binds_the_conforming_pure_bounded_u_family(self) -> None:
        contract = load_yaml_subset(
            EPISODE_ROOT
            / "outputs"
            / "executor"
            / "SCIENTIFIC_CONTRACT_POST36_V3.yaml"
        )
        _require_active_model_family_conformance()
        settings = _contract_settings(contract)
        self.assertEqual(settings["integration_draws"], 2048)
        self.assertEqual(
            contract["models"]["U_linear"]["latent_support_geometry"],
            "rank_at_most_2_Gaussian_manifold_in_ILR",
        )
        self.assertFalse(
            contract["models"]["U_curved"]["ambient_residual_tube"]
        )
        self.assertEqual(
            contract["models"]["M"]["within_group_continuous_law"],
            "same_rank_at_most_2_Gaussian_manifold_as_U_linear",
        )
        self.assertTrue(
            contract["execution_parameters"][
                "predictive_and_fitted_null_use_identical_stored_latent_support_and_weights"
            ]
        )
        with mock.patch(
            "ep12_executor.scientific_qualification.BOUNDED_LATENT_SUPPORT_PROTOCOL",
            "unsupported",
        ):
            with self.assertRaisesRegex(ValueError, "implementation protocol"):
                _require_active_model_family_conformance()


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock numpy/scipy/scikit-learn module environment",
)
class ScientificImplementationTests(unittest.TestCase):
    def test_development_endpoint_mapping_and_pooling_preserve_mass(self) -> None:
        self.assertEqual(_endpoint_bin("missing_annotation"), "missing_annotation")
        self.assertEqual(
            _endpoint_bin("untyped_status::Orphan::Orphan-artifact"), "fragment"
        )
        self.assertEqual(
            _endpoint_bin("untyped_status::Anchor::Soma Anchor"), "proofreading"
        )
        self.assertEqual(
            _endpoint_bin("untyped_status::Traced::Reviewed"), "untyped"
        )
        raw = [
            "typed::common",
            "typed::rare",
            "untyped_status::Orphan::Orphan",
            "missing_annotation",
        ]
        matrix = sparse.csr_matrix(
            np.array(
                [
                    [10, 1, 2, 0],
                    [8, 0, 0, 1],
                    [9, 0, 3, 0],
                    [7, 0, 0, 1],
                ],
                dtype=np.int64,
            )
        )
        collapsed, vocabulary, record = _collapse_vocabulary(
            matrix,
            np.array(["T1", "T1", "T2", "T2"]),
            raw,
            strategy="development_frequency_pooling",
            minimum_development_type_count=2,
            rare_partner_mass_fraction=0.01,
        )
        self.assertIn("typed::common", vocabulary)
        self.assertNotIn("typed::rare", vocabulary)
        self.assertEqual(record["retained_typed_partner_count"], 1)
        np.testing.assert_array_equal(
            np.asarray(collapsed.sum(axis=1)).ravel(),
            np.asarray(matrix.sum(axis=1)).ravel(),
        )

    def test_ilr_basis_is_orthonormal_and_sum_zero(self) -> None:
        basis = helmert_ilr_basis(6)
        np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(basis.sum(axis=0), np.zeros(5), atol=1e-12)

    def test_generator_retains_integer_counts_totals_and_unknown_mass(self) -> None:
        self.assertEqual(
            set(PARTNER_NAMES[5:]),
            {
                "explicit_unknown",
                "untyped",
                "fragment",
                "proofreading",
                "missing_annotation",
                "out_of_vocabulary",
            },
        )
        fixture = generate_synthetic_type(
            "side_specific_endpoint_loss",
            left_neurons=8,
            right_neurons=8,
            count_total=100,
            strength=1.0,
            missing_fraction=0.30,
            seed=12,
        )
        for side in (fixture.left, fixture.right):
            counts, covariates = validate_profile_arrays(side.counts, side.covariates)
            self.assertTrue(np.issubdtype(counts.dtype, np.integer))
            self.assertTrue(np.all(counts.sum(axis=1) > 0))
            self.assertEqual(covariates.shape, (8, 2))
        self.assertGreater(fixture.right.endpoint_loss_fraction.mean(), 0.0)
        self.assertEqual(fixture.left.endpoint_loss_fraction.mean(), 0.0)
        unknown_index = fixture.partner_names.index("explicit_unknown")
        self.assertGreater(
            np.mean(
                fixture.right.counts[:, unknown_index]
                / fixture.right.counts.sum(axis=1)
            ),
            np.mean(
                fixture.left.counts[:, unknown_index]
                / fixture.left.counts.sum(axis=1)
            ),
        )

    def test_target_scoring_is_deterministic_and_does_not_refit(self) -> None:
        fixture = generate_synthetic_type(
            "residual_groups_within_continuum",
            left_neurons=12,
            right_neurons=9,
            count_total=80,
            strength=1.8,
            missing_fraction=0.0,
            seed=91,
        )
        self.assertEqual(
            len(np.unique(fixture.left.group)),
            len(np.unique(fixture.right.group)),
        )
        self.assertEqual(
            len(np.unique(fixture.left.group)),
            fixture.generator_parameters["group_components"],
        )
        model = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "M2",
            seed=13,
            integration_draws=8,
        )
        coefficients_before = model.nuisance_coefficients.copy()
        means_before = model.prior_means.copy()
        first = model.score_neurons(fixture.right.counts, fixture.right.covariates)
        second = model.score_neurons(fixture.right.counts, fixture.right.covariates)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(model.nuisance_coefficients, coefficients_before)
        np.testing.assert_array_equal(model.prior_means, means_before)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_default_integration_seed_preserves_seed_driven_behavior(self) -> None:
        fixture = generate_synthetic_type(
            "linear_continuum",
            left_neurons=8,
            right_neurons=6,
            count_total=50,
            strength=1.0,
            missing_fraction=0.0,
            seed=313,
        )
        implicit = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "U_linear",
            seed=314,
            integration_draws=8,
        )
        explicit = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "U_linear",
            seed=314,
            integration_seed=314,
            integration_draws=8,
        )
        self.assertEqual(implicit.integration_seed, 314)
        np.testing.assert_array_equal(
            implicit.latent_offsets, explicit.latent_offsets
        )
        np.testing.assert_array_equal(
            implicit.score_neurons(fixture.right.counts, fixture.right.covariates),
            explicit.score_neurons(fixture.right.counts, fixture.right.covariates),
        )

    def test_explicit_integration_seed_retains_nested_qmc_prefix(self) -> None:
        fixture = generate_synthetic_type(
            "linear_continuum",
            left_neurons=8,
            right_neurons=6,
            count_total=50,
            strength=1.0,
            missing_fraction=0.0,
            seed=317,
        )
        models = [
            fit_predictive_model(
                fixture.left.counts,
                fixture.left.covariates,
                "U_linear",
                seed=318,
                integration_seed=319,
                integration_draws=draws,
            )
            for draws in (8, 16)
        ]
        self.assertEqual(models[0].integration_seed, models[1].integration_seed)
        np.testing.assert_array_equal(
            models[0].latent_offsets, models[1].latent_offsets[:8]
        )

    def test_cv_candidates_share_fold_integration_not_fit_seeds(self) -> None:
        counts = np.array(
            [
                [8, 2, 1],
                [7, 3, 1],
                [6, 4, 1],
                [3, 7, 1],
                [2, 8, 1],
                [4, 6, 1],
            ],
            dtype=np.int64,
        )
        covariates = np.zeros((len(counts), 1), dtype=float)
        observed: list[dict[str, object]] = []

        def record_fit(
            training_counts: np.ndarray,
            training_covariates: np.ndarray,
            model_id: str,
            **settings: object,
        ) -> SimpleNamespace:
            del training_counts, training_covariates
            observed.append(
                {
                    "model_id": model_id,
                    "fit_seed": settings["seed"],
                    "integration_seed": settings["integration_seed"],
                }
            )
            parameter_count = 2 if model_id == "U_linear" else 3
            return SimpleNamespace(
                parameter_count=parameter_count,
                mean_score=lambda *_args: 0.0,
            )

        with mock.patch(
            "ep12_executor.scientific_models.fit_predictive_model",
            side_effect=record_fit,
        ):
            cross_validated_candidate(
                counts,
                covariates,
                ["U_linear", "U_curved"],
                seed=315,
                folds=3,
                integration_draws=8,
            )

        by_model = {
            model_id: [row for row in observed if row["model_id"] == model_id]
            for model_id in ("U_linear", "U_curved")
        }
        for fold in range(3):
            linear = by_model["U_linear"][fold]
            curved = by_model["U_curved"][fold]
            self.assertEqual(
                linear["integration_seed"], curved["integration_seed"]
            )
            self.assertEqual(
                linear["integration_seed"],
                seed_from(315, "cv-integration", fold),
            )
            self.assertNotEqual(linear["fit_seed"], curved["fit_seed"])
        self.assertEqual(len({row["fit_seed"] for row in observed}), 6)

    def test_final_triplet_models_share_integration_not_fit_seeds(self) -> None:
        counts = np.array(
            [
                [8, 2, 1],
                [7, 3, 1],
                [6, 4, 1],
                [3, 7, 1],
                [2, 8, 1],
                [4, 6, 1],
            ],
            dtype=np.int64,
        )
        covariates = np.zeros((len(counts), 1), dtype=float)
        observed: list[dict[str, object]] = []

        def select_candidate(
            source_counts: np.ndarray,
            source_covariates: np.ndarray,
            candidates: list[str],
            **settings: object,
        ) -> CandidateSelection:
            del source_counts, source_covariates, settings
            selected = candidates[0]
            mean_score = 1.0 if selected.startswith("U") else 0.0
            return CandidateSelection(
                model_id=selected,
                mean_score=mean_score,
                standard_error=0.0,
                fold_scores=(mean_score,),
                parameter_count=1,
            )

        def record_final_fit(
            source_counts: np.ndarray,
            source_covariates: np.ndarray,
            model_id: str,
            **settings: object,
        ) -> SimpleNamespace:
            del source_counts, source_covariates
            observed.append(
                {
                    "model_id": model_id,
                    "fit_seed": settings["seed"],
                    "integration_seed": settings["integration_seed"],
                }
            )
            return SimpleNamespace(
                model_id=model_id,
                integration_seed=settings["integration_seed"],
            )

        with mock.patch(
            "ep12_executor.scientific_models.cross_validated_candidate",
            side_effect=select_candidate,
        ), mock.patch(
            "ep12_executor.scientific_models.fit_predictive_model",
            side_effect=record_final_fit,
        ):
            triplet = fit_triplet(counts, covariates, seed=316)

        expected_integration_seed = seed_from(316, "final-integration")
        self.assertEqual(
            {row["integration_seed"] for row in observed},
            {expected_integration_seed},
        )
        self.assertEqual(len({row["fit_seed"] for row in observed}), 3)
        self.assertEqual(
            {
                triplet.T.integration_seed,
                triplet.U.integration_seed,
                triplet.M.integration_seed,
            },
            {expected_integration_seed},
        )

    def test_declared_representation_and_covariance_branches_execute(self) -> None:
        fixture = generate_synthetic_type(
            "linear_continuum",
            left_neurons=10,
            right_neurons=8,
            count_total=60,
            strength=1.0,
            missing_fraction=0.0,
            seed=211,
        )
        for representation in (
            "count_aware_log_ratio",
            "low_rank_composition",
            "low_rank_count_model",
        ):
            for covariance in ("diagonal", "shrinkage", "low_rank"):
                model = fit_predictive_model(
                    fixture.left.counts,
                    fixture.left.covariates,
                    "U_linear",
                    seed=212,
                    integration_draws=4,
                    latent_rank=2,
                    pseudocount=0.1,
                    representation=representation,
                    covariance_mode=covariance,
                    covariance_rank=2,
                )
                scores = model.score_neurons(
                    fixture.right.counts, fixture.right.covariates
                )
                self.assertTrue(np.all(np.isfinite(scores)))
                self.assertEqual(model.diagnostics["representation"], representation)
                self.assertEqual(model.diagnostics["covariance_mode"], covariance)
                centered = model.latent_offsets - model.latent_offsets.mean(
                    axis=0, keepdims=True
                )
                self.assertLessEqual(np.linalg.matrix_rank(centered, tol=1e-10), 2)
                self.assertLessEqual(
                    np.linalg.matrix_rank(model.prior_covariances[0], tol=1e-10),
                    2,
                )
                self.assertFalse(model.diagnostics["ambient_residual_tube"])
                self.assertEqual(
                    model.diagnostics["latent_support_protocol"],
                    BOUNDED_LATENT_SUPPORT_PROTOCOL,
                )

    def test_curved_reference_supports_fewer_draws_than_quadrature_nodes(self) -> None:
        fixture = generate_synthetic_type(
            "curved_continuum_dense",
            left_neurons=9,
            right_neurons=7,
            count_total=60,
            strength=1.2,
            missing_fraction=0.0,
            seed=31,
        )
        model = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "U_curved",
            seed=32,
            integration_draws=4,
        )
        scores = model.score_neurons(fixture.right.counts, fixture.right.covariates)
        self.assertEqual(len(scores), 7)
        self.assertTrue(np.all(np.isfinite(scores)))
        quadrature_z, quadrature_weights = np.polynomial.hermite_e.hermegauss(9)
        quadrature_weights = quadrature_weights / np.sqrt(2.0 * np.pi)
        quadrature_weights /= quadrature_weights.sum()
        curve = np.asarray(model.diagnostics["curve_coefficients"], dtype=float)
        np.testing.assert_allclose(
            model.latent_offsets,
            _cubic_hermite_basis(quadrature_z) @ curve,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.exp(model.log_weights), quadrature_weights, atol=1e-15, rtol=0.0
        )
        np.testing.assert_array_equal(
            model.prior_covariances,
            np.zeros_like(model.prior_covariances),
        )
        self.assertEqual(model.diagnostics["latent_dimension"], 1)
        self.assertEqual(model.diagnostics["curve_degree"], 3)
        self.assertEqual(model.diagnostics["integration_nodes"], 9)
        self.assertFalse(model.diagnostics["ambient_residual_tube"])

    def test_scoring_uses_exact_stored_support_and_weights(self) -> None:
        fixture = generate_synthetic_type(
            "linear_continuum",
            left_neurons=10,
            right_neurons=8,
            count_total=60,
            strength=1.0,
            missing_fraction=0.0,
            seed=401,
        )
        model = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "U_linear",
            seed=402,
            integration_draws=8,
        )
        baseline = model.score_neurons(
            fixture.right.counts, fixture.right.covariates
        )
        prior_only_change = replace(
            model,
            prior_means=np.full_like(model.prior_means, 100.0),
            prior_covariances=np.zeros_like(model.prior_covariances),
        )
        np.testing.assert_array_equal(
            baseline,
            prior_only_change.score_neurons(
                fixture.right.counts, fixture.right.covariates
            ),
        )
        support_change = replace(
            model,
            latent_offsets=model.latent_offsets + 2.0,
        )
        self.assertFalse(
            np.allclose(
                baseline,
                support_change.score_neurons(
                    fixture.right.counts, fixture.right.covariates
                ),
                atol=1e-10,
                rtol=0.0,
            )
        )

    def test_m_uses_same_rank_bounded_within_group_law_plus_offsets(self) -> None:
        fixture = generate_synthetic_type(
            "residual_groups_within_continuum",
            left_neurons=14,
            right_neurons=10,
            count_total=80,
            strength=1.8,
            missing_fraction=0.0,
            seed=411,
        )
        model = fit_predictive_model(
            fixture.left.counts,
            fixture.left.covariates,
            "M2",
            seed=412,
            integration_draws=8,
            latent_rank=2,
        )
        self.assertTrue(model.diagnostics["group_offsets"])
        self.assertEqual(
            model.diagnostics["within_group_latent_support_family"],
            "rank_bounded_Gaussian_manifold",
        )
        self.assertLessEqual(
            model.diagnostics["within_group_latent_support_rank"], 2
        )
        for covariance in model.prior_covariances:
            self.assertLessEqual(np.linalg.matrix_rank(covariance, tol=1e-10), 2)
        per_component = len(model.latent_offsets) // len(model.prior_means)
        for component, mean in enumerate(model.prior_means):
            support = model.latent_offsets[
                component * per_component : (component + 1) * per_component
            ]
            self.assertLessEqual(
                np.linalg.matrix_rank(support - mean[None, :], tol=1e-10), 2
            )

    def test_retaining_count_totals_changes_predictive_precision(self) -> None:
        source = np.array(
            [
                [8, 2, 1],
                [7, 3, 1],
                [2, 8, 1],
                [3, 7, 1],
                [6, 4, 1],
                [4, 6, 1],
            ],
            dtype=np.int64,
        )
        covariates = np.zeros((len(source), 1))
        model = fit_predictive_model(
            source,
            covariates,
            "U_linear",
            seed=5,
            integration_draws=16,
        )
        low = np.array([[8, 2, 1], [8, 2, 1]], dtype=np.int64)
        high = low * 100
        target_covariates = np.zeros((2, 1))
        low_score = model.score_neurons(low, target_covariates)
        high_score = model.score_neurons(high, target_covariates)
        self.assertFalse(np.allclose(low_score, high_score, atol=1e-5, rtol=0.0))

    def test_simultaneous_interval_and_exact_binomial_are_computed(self) -> None:
        values = np.array(
            [[0.010, 0.012], [0.018, 0.014], [0.021, 0.019], [0.013, 0.017]]
        )
        interval = paired_simultaneous_intervals(
            values, margin=0.02, replicates=99, seed=3
        )
        self.assertEqual(interval["labels"], ["left_to_right", "right_to_left", "aggregate"])
        self.assertEqual(len(interval["lower"]), 3)
        self.assertTrue(np.all(np.asarray(interval["lower"]) <= np.asarray(interval["upper"])))
        lower, upper = exact_binomial_interval(0, 72)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_leave_one_replicate_out_coverage_fails_closed(self) -> None:
        cell = QualificationCell(
            cell_id="adequate_template__test",
            scenario="adequate_template",
            neurons_per_side=7,
            count_total=100,
            strength=1.8,
            missing_fraction=0.1,
            acceptance_role="core_operating_characteristic",
        )
        records = [
            {
                "terminal": "unresolved",
                "intervals": {
                    "estimate": [0.0, 0.0, 0.0],
                    "lower": [-0.1, -0.1, -0.1],
                    "upper": [0.1, 0.1, 0.1],
                },
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
            },
            {
                "terminal": "unresolved",
                "intervals": {
                    "estimate": [0.01, 0.01, 0.01],
                    "lower": [-0.1, -0.1, -0.1],
                    "upper": [0.1, 0.1, 0.1],
                },
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
            },
        ]
        summary = _cell_summary(cell, records)
        coverage = summary["empirical_interval_coverage"]
        self.assertFalse(coverage["reference_centers_independent"])
        self.assertFalse(coverage["coverage_indicators_independent"])
        self.assertFalse(coverage["exact_binomial_acceptance_valid"])
        passed, reasons = _accept_cell(
            summary,
            {
                "false_M_rate_max": 0.05,
                "false_adequacy_rate_max": 0.05,
                "interval_nominal_coverage": 0.95,
            },
        )
        self.assertFalse(passed)
        self.assertIn(
            "coverage_estimand_or_indicator_independence_not_qualified", reasons
        )


if __name__ == "__main__":
    unittest.main()
