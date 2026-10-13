from __future__ import annotations

import hashlib
import inspect
import unittest
from dataclasses import replace

try:
    import numpy as np
    from scipy.special import gammaln

    import ep12_executor.scientific_oracle as oracle_module
    from ep12_executor.scientific_oracle import (
        OracleError,
        OracleScientificChoiceRequired,
        classify_directional_margin_interval,
        population_kl_group_gain,
        true_generator_log_predictive_density,
    )
    from ep12_executor.scientific_synthetic import (
        generator_probabilities,
        generate_synthetic_type,
        generate_synthetic_type_with_dgp,
    )

    SCIENTIFIC_STACK_AVAILABLE = True
except (ImportError, RuntimeError):
    SCIENTIFIC_STACK_AVAILABLE = False


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "needs the pinned Sherlock NumPy/SciPy module environment",
)
class SyntheticOracleTests(unittest.TestCase):
    REPLAY_CASES = (
        (
            "adequate_template",
            "f7fbbc8ff148cbe8c4485ab1b1b56142d962d380172f1c3154322cde2e91da92",
            "db7c708da34b54ae17321dfcc875083e810e3d0282343f0a100515cac3b6ce97",
        ),
        (
            "linear_continuum",
            "984258a3c60dd6dd93b82c45aa4056efade4941d5449aa38d80b877966ca5d8f",
            "86ab56b7fe184d66d9b070767629b7420279779c1a9df33cf74a40ef3ba43262",
        ),
        (
            "curved_continuum_dense",
            "8234790b50bcf64c833356119bc793c841f6efe75fd6f3eb1588e5a3811f287d",
            "46bf78ce4ba0eb80af42024f7ac3de6c58102971ae045c4ccfab19e16b04d414",
        ),
        (
            "curved_continuum_sparse_intermediates",
            "726346f69ecf6def7c7f7f839be46d5ace069778f154afddc1855a8adb2d7f35",
            "1bf0966222ebc41554e7b81ab78b91396cef2d4425b61e5ed61532d2ff266e5d",
        ),
        (
            "residual_groups_within_continuum",
            "336c6ae0da235135138591139eadd2d286b74a9e70611f5b6d50a2c376443bb1",
            "f72badba14187dd8c6c9156b89312187b8695350ae1bac44bbc69e237bb0569d",
        ),
        (
            "weak_residual_groups",
            "7801ecae003292c7ba0791efb2009c534e6f99cc95820f3e444684a5d95cf140",
            "27c11c44d58fd4c754c44bfd8a32e7b5a149389111d01c8f722ad9cc8bfb4798",
        ),
        (
            "side_specific_endpoint_loss",
            "e0de998b5f75ebcbe36e539ac18c0e3ccbd5c51da212831f76685e187f221939",
            "3f492c1024e67b4e8d399bba8390bf88e97e6a00c586c013d6f4a2575fafe31b",
        ),
        (
            "anatomy_quality_confound",
            "843cad5025a95e3df17ce4acb8325058e21115418f601e414e6a735ea6fc0887",
            "4e5ee5100d47de8328c71c593f00aa16f04b540d0295b89949fd4278d33381bc",
        ),
        (
            "high_strength_or_influential_cells",
            "af0948c61e4c8e6ffca98daf932f140a9de711b4798af213a54a1d9401c68c06",
            "4c7c96bd41c416312373030396685093e28197c7ce7408b29e382a2cadc0d70b",
        ),
    )

    @staticmethod
    def _bundle(scenario: str, seed: int = 91):
        return generate_synthetic_type_with_dgp(
            scenario,
            left_neurons=8,
            right_neurons=7,
            count_total=100,
            strength=1.8,
            missing_fraction=0.3,
            seed=seed,
        )

    def test_refactor_replays_all_historical_fixture_branches_exactly(self) -> None:
        for offset, (scenario, left_hash, right_hash) in enumerate(
            self.REPLAY_CASES
        ):
            with self.subTest(scenario=scenario):
                fixture = generate_synthetic_type(
                    scenario,
                    left_neurons=8,
                    right_neurons=7,
                    count_total=100,
                    strength=1.8,
                    missing_fraction=0.3,
                    seed=1000 + offset,
                )
                self.assertEqual(
                    hashlib.sha256(fixture.left.counts.tobytes()).hexdigest(),
                    left_hash,
                )
                self.assertEqual(
                    hashlib.sha256(fixture.right.counts.tobytes()).hexdigest(),
                    right_hash,
                )

    def test_shared_probability_path_reconstructs_realized_fixture_law(self) -> None:
        for offset, (scenario, _, _) in enumerate(self.REPLAY_CASES):
            with self.subTest(scenario=scenario):
                bundle = self._bundle(scenario, seed=2000 + offset)
                for side_name in ("left", "right"):
                    side = getattr(bundle.fixture, side_name)
                    state = getattr(bundle, f"{side_name}_dgp")
                    reconstructed = []
                    for index, latent_coordinate in enumerate(
                        side.latent_coordinate
                    ):
                        center = None
                        if state.latent_family == "grouped_normal_linear":
                            center = state.group_centers[side.group[index]]
                        reconstructed.append(
                            generator_probabilities(
                                state,
                                index,
                                latent_coordinate=float(latent_coordinate),
                                group_center=(
                                    None if center is None else float(center)
                                ),
                            )
                        )
                    np.testing.assert_allclose(
                        np.vstack(reconstructed),
                        state.realized_probabilities,
                        atol=1e-15,
                        rtol=0.0,
                    )
                    np.testing.assert_array_equal(
                        side.counts.sum(axis=1), state.count_totals
                    )

    def test_point_mass_oracle_matches_direct_multinomial_score(self) -> None:
        bundle = self._bundle("adequate_template")
        side = bundle.fixture.right
        state = bundle.right_dgp
        observed = true_generator_log_predictive_density(
            side.counts, state, quadrature_nodes=3
        )
        expected = []
        for row, probability in zip(side.counts, state.realized_probabilities):
            total = int(row.sum())
            log_probability = (
                gammaln(total + 1.0)
                - np.sum(gammaln(row.astype(float) + 1.0))
                + np.log(probability) @ row.astype(float)
            )
            expected.append(float(log_probability) / total)
        np.testing.assert_allclose(
            observed.per_neuron_log_predictive_density_per_count,
            np.asarray(expected),
            atol=1e-14,
            rtol=0.0,
        )
        self.assertEqual(observed.target_group, "marginalized")
        self.assertEqual(observed.target_latent_coordinate, "marginalized")
        self.assertFalse(observed.fitted_qualification_results_used)

    def test_group_oracle_marginal_is_label_permutation_invariant(self) -> None:
        bundle = self._bundle("residual_groups_within_continuum")
        side = bundle.fixture.right
        state = bundle.right_dgp
        first = true_generator_log_predictive_density(
            side.counts, state, quadrature_nodes=9
        )
        permutation = np.arange(state.group_components)[::-1]
        permuted = replace(
            state,
            group_centers=state.group_centers[permutation],
            group_weights=state.group_weights[permutation],
        )
        second = true_generator_log_predictive_density(
            side.counts, permuted, quadrature_nodes=9
        )
        np.testing.assert_allclose(
            first.per_neuron_log_predictive_density_per_count,
            second.per_neuron_log_predictive_density_per_count,
            atol=1e-14,
            rtol=0.0,
        )

    def test_margin_classification_uses_only_supplied_bounds_and_margin(self) -> None:
        self.assertEqual(
            classify_directional_margin_interval(
                [0.021, 0.022], [0.030, 0.031], margin=0.02
            ),
            "above_margin",
        )
        self.assertEqual(
            classify_directional_margin_interval(
                [0.001, 0.002], [0.019, 0.018], margin=0.02
            ),
            "below_margin",
        )
        self.assertEqual(
            classify_directional_margin_interval(
                [0.019, 0.021], [0.025, 0.026], margin=0.02
            ),
            "margin_indeterminate",
        )
        self.assertEqual(
            classify_directional_margin_interval(
                [0.020, 0.021], [0.025, 0.026], margin=0.02
            ),
            "margin_indeterminate",
        )

    def test_population_kl_gain_fails_closed_at_unfrozen_u_projection(self) -> None:
        state = self._bundle("residual_groups_within_continuum").left_dgp
        with self.assertRaisesRegex(
            OracleScientificChoiceRequired, "freeze the source-side population T/U"
        ):
            population_kl_group_gain(state=state)
        nongroup = self._bundle("linear_continuum").left_dgp
        with self.assertRaisesRegex(OracleError, "requires a group generator"):
            population_kl_group_gain(state=nongroup)

    def test_oracle_module_has_no_fitted_model_dependency(self) -> None:
        source = inspect.getsource(oracle_module)
        self.assertNotIn("scientific_models", source)
        self.assertNotIn("reciprocal_transfer", source)


if __name__ == "__main__":
    unittest.main()
