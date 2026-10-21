"""Generated-only canonical null, serialization, and observed/replay parity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.canonical_target import CANONICAL_ENDPOINTS, DECODER_PROTOCOL
from ep12_executor.canonical_view import pool_counts, prepare_canonical_view
from ep12_executor.fitted_null import (
    CANONICAL_SAMPLER_PROTOCOL, TypePooledExpansionPlan, _stable_seed,
    _typed_probabilities, freeze_null_seed_manifest, null_seed_binding, sample_fitted_null,
)
from ep12_executor.full_search_runtime import build_selected_fitted_null_state
from ep12_executor.model_artifacts import load_model_state, write_model_state
from ep12_executor.null_trial_runner import NullTrialInputs, ProductionFullSearchCallback
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.runtime import EPISODE_ROOT


class CanonicalNullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
        cls.config = dict(coverage_configurations(cls.policy)[0])
        cls.config.update(nuisance_spline_df=0, component_count=2,
                          partner_vocabulary="development_frequency_pooling",
                          partner_minimum_development_type_count=2,
                          rare_partner_pooling_mass_fraction=.01)
        cls.names = ["typed::a", "typed::b", "typed::c", "typed::d"] + list(CANONICAL_ENDPOINTS)
        cls.hierarchy = {name: {"class": "shared", "subclass": name} for name in "abcd"}
        rng = np.random.default_rng(20261021)
        cls.raw = np.vstack([
            rng.multinomial(80, [.36, .04, .15, .15, .05, .05, .05, .05, .05, .05], size=12),
            rng.multinomial(80, [.04, .36, .15, .15, .05, .05, .05, .05, .05, .05], size=12),
        ])
        cls.types = np.array(["focal"] * 24)
        cls.sides = np.array(["L"] * 12 + ["R"] * 12)
        cls.covariates = {df: np.zeros((24, 22 + (0 if df == 0 else 6 * (df - 1)))) for df in (0, 3, 5)}
        cls.design, cls.generators, cls.filtered = build_selected_fitted_null_state(
            policy=cls.policy, neuron_ids=list(range(24)), focal_types=cls.types,
            sides=cls.sides, raw_counts=cls.raw, raw_vocabulary=cls.names,
            partner_hierarchy=cls.hierarchy, covariates_by_spline_df=cls.covariates,
            selected_configuration=cls.config,
            fit_settings={"integration_draws": 16, "source_cv_folds": 2},
        )
        cls.expansion = TypePooledExpansionPlan.build(cls.design)
        cls.binding = null_seed_binding(cls.design, cls.generators, cls.expansion,
                                       selection_sha256="ab" * 32)
        cls.seeds = freeze_null_seed_manifest(master_seed=27, binding=cls.binding)

    def sample(self):
        return sample_fitted_null(self.design, self.generators, self.expansion,
                                  seed_manifest=self.seeds, replicate_index=0,
                                  selection_sha256="ab" * 32)

    def test_sampler_uses_stored_canonical_law_and_varies_endpoints(self):
        sample = self.sample()
        self.assertEqual(self.binding["sampler_protocol"], CANONICAL_SAMPLER_PROTOCOL)
        self.assertEqual(sample.score_target_protocol, DECODER_PROTOCOL)
        self.assertFalse(self.expansion.rules)  # no both-side expansion estimate
        np.testing.assert_array_equal(sample.raw_counts.sum(axis=1), self.raw.sum(axis=1))
        self.assertTrue(np.any(sample.raw_counts[:, 4:] != self.raw[:, 4:]))
        key = (self.design.focal_types[0], self.design.sides[0])
        probabilities = _typed_probabilities(self.generators[key], self.design, 0,
                                             sample.replicate_seed, canonical=True)
        seed = _stable_seed(CANONICAL_SAMPLER_PROTOCOL, sample.replicate_seed,
                            self.design.neuron_ids[0], *key, "canonical_multinomial")
        expected = np.random.default_rng(seed).multinomial(80, probabilities)
        np.testing.assert_array_equal(sample.raw_counts[0], expected)
        np.testing.assert_array_equal(self.sample().raw_counts, sample.raw_counts)

    def test_decoder_is_source_side_specific_and_survives_artifact_roundtrip(self):
        left = self.generators[("focal", "L")].canonical_decoder
        right = self.generators[("focal", "R")].canonical_decoder
        self.assertGreater(np.exp(left.log_conditional_probability[0]),
                           np.exp(right.log_conditional_probability[0]))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "state"
            write_model_state(path, design=self.design, generators=self.generators)
            design, generators, _ = load_model_state(path)
            self.assertEqual(design.canonical_target_protocol, DECODER_PROTOCOL)
            for key in self.generators:
                np.testing.assert_array_equal(
                    generators[key].canonical_decoder.log_conditional_probability,
                    self.generators[key].canonical_decoder.log_conditional_probability,
                )
            expansion = TypePooledExpansionPlan.build(design)
            actual = sample_fitted_null(design, generators, expansion,
                                        seed_manifest=self.seeds, replicate_index=0,
                                        selection_sha256="ab" * 32)
            np.testing.assert_array_equal(actual.raw_counts, self.sample().raw_counts)

    def test_generated_primary_replay_scores_canonical_targets(self):
        with tempfile.TemporaryDirectory() as name:
            callback = ProductionFullSearchCallback(
                policy=self.policy, inputs=NullTrialInputs(
                    raw_vocabulary=tuple(self.names), covariates_by_spline_df=self.filtered,
                    partner_hierarchy=self.hierarchy,
                    fit_settings={"integration_draws": 16, "source_cv_folds": 2},
                    artifact_root=Path(name), callback_data_sha256="ac" * 32,
                ),
            )
            trial = {"configuration": self.config, "configuration_sha256": digest_object(self.config)}
            result = callback._base_fit(self.sample(), trial)
            self.assertEqual(result["score_target_protocol"], DECODER_PROTOCOL)
            self.assertEqual(result["valid_type_count"], 1)
            self.assertTrue(np.isfinite(result["objective"]["bilateral"]))

    def test_raw_status_aliases_and_canonical_views_have_identical_controls(self):
        # Two raw status keys share the same semantic endpoint.
        names = self.names[:4] + ["untyped_status::Traced::x", "untyped_status::Traced::y", "missing_annotation"]
        raw = np.column_stack((self.raw[:, :4], self.raw[:, 4], self.raw[:, 5], self.raw[:, 8]))
        baseline = prepare_canonical_view(raw, self.types, names, self.config, self.hierarchy)
        canonical_names = self.names
        cases = [dict(), dict(matrix_transform="binary"),
                 dict(matrix_transform="pooled_multinomial_noise", transform_seed=55),
                 dict(selected_keys=["typed::a", "typed::b"]), dict(endpoint="untyped")]
        for options in cases:
            with self.subTest(options=options):
                observed = prepare_canonical_view(raw, self.types, names, self.config, self.hierarchy, **options)
                generated = prepare_canonical_view(baseline[1], self.types, canonical_names,
                                                   self.config, self.hierarchy, **options)
                np.testing.assert_array_equal(observed[0].toarray(), generated[0].toarray())
                np.testing.assert_array_equal(observed[1].toarray(), generated[1].toarray())
                np.testing.assert_array_equal(observed[2], generated[2])
                rebuilt = pool_counts(observed[1], observed[2], observed[0].shape[1])
                np.testing.assert_array_equal(rebuilt.toarray(), observed[0].toarray())
                self.assertEqual(set(observed[2]), set(range(observed[0].shape[1])))
                if options.get("matrix_transform") != "binary":
                    np.testing.assert_array_equal(observed[1].sum(axis=1), raw.sum(axis=1)[:, None])

    def test_generated_datasets_recompute_outcome_dependent_pooling(self):
        config = {**self.config, "rare_partner_pooling_mass_fraction": .25}
        types = np.array(["a", "a", "b", "b"])
        first = np.array([[8, 1, 1, 0]] * 4)
        second = np.array([[1, 8, 1, 0]] * 4)
        one = prepare_canonical_view(first, types, self.names[:4], config, self.hierarchy)
        two = prepare_canonical_view(second, types, self.names[:4], config, self.hierarchy)
        self.assertIn("typed::a", one[3])
        self.assertNotIn("typed::a", two[3])
        self.assertIn("typed::b", two[3])


if __name__ == "__main__":
    unittest.main()
