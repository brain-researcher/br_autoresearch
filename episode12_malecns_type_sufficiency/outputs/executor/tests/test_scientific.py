from __future__ import annotations

import unittest

from ep12_executor.runtime import EPISODE_ROOT
from ep12_executor.yaml_subset import load_yaml_subset

try:
    import numpy as np
    from scipy import sparse

    from ep12_executor.development_trial import _collapse_vocabulary, _endpoint_bin
    from ep12_executor.scientific_models import (
        fit_predictive_model,
        helmert_ilr_basis,
        validate_profile_arrays,
    )
    from ep12_executor.scientific_qualification import (
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


if __name__ == "__main__":
    unittest.main()
