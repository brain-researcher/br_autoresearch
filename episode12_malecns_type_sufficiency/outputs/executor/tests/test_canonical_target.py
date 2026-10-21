"""Focused generated-only checks; never open a MaleCNS snapshot or result."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp

from ep12_executor.canonical_target import (
    CANONICAL_ENDPOINTS, DECODER_PROTOCOL, CanonicalDecoder, canonicalize_counts,
)
from ep12_executor.development_trial import _collapse_vocabulary
from ep12_executor.production_readiness import (
    ReadinessError, evaluate_production_readiness, verify_contract_roles,
    verify_observed_provisional_chain,
)
from ep12_executor.runtime import EPISODE_ROOT
from ep12_executor.scientific_models import (
    cross_validated_candidate, fit_predictive_model, fit_triplet,
    reciprocal_transfer, seed_from,
)


class CanonicalTargetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261020)
        self.mapping = np.array([0, 0, 1, 1, 2, 2])
        self.left = rng.multinomial(40, [.35, .05, .2, .1, .25, .05], size=16)
        self.right = rng.multinomial(40, [.05, .35, .1, .2, .05, .25], size=16)
        self.covariates = np.zeros((16, 1))
        self.decoder = CanonicalDecoder.fit(self.left, self.mapping, 3)
        self.pooled_left = self.decoder.pooled_counts(self.left).astype(np.int64)
        self.pooled_right = self.decoder.pooled_counts(self.right).astype(np.int64)

    def test_factorization_matches_direct_integrated_canonical_likelihood(self):
        for family in ("T", "U_linear", "U_curved", "M2"):
            with self.subTest(family=family):
                model = fit_predictive_model(
                    self.pooled_left, self.covariates, family,
                    seed=92, integration_seed=93, integration_draws=16,
                )
                pooled_scores = model.score_neurons(self.pooled_right, self.covariates)
                corrected = pooled_scores + self.decoder.correction(self.right, self.pooled_right)
                offsets, weights = model.validated_latent_support()
                base = np.column_stack((np.ones(16), self.covariates)) @ model.nuisance_coefficients
                direct = []
                for row, mean in zip(self.right, base):
                    logits = (mean[None, :] + offsets) @ model.basis.T
                    pooled_p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
                    canonical_p = self.decoder.decode_probabilities(pooled_p)
                    ll = gammaln(row.sum() + 1) - gammaln(row + 1).sum() + np.log(canonical_p) @ row
                    direct.append(logsumexp(ll + weights) / row.sum())
                np.testing.assert_allclose(corrected, direct, atol=1e-12, rtol=0)

    def test_zero_observed_categories_receive_fixed_per_category_prior(self):
        decoder = CanonicalDecoder.fit([[8, 0, 0, 0, 0]], [0, 0, 1, 1, 2], 3)
        np.testing.assert_allclose(np.exp(decoder.log_conditional_probability),
                                   [8.5 / 9, .5 / 9, .5, .5, 1.0])
        decoded = decoder.decode_probabilities([[.2, .3, .5]])
        self.assertAlmostEqual(decoded.sum(), 1.0)
        with self.assertRaisesRegex(ValueError, "structurally empty"):
            CanonicalDecoder.fit([[2, 3]], [0, 0], 2)

    def test_mapping_preserves_all_endpoint_mass_and_removes_only_empty_pool(self):
        raw = ["typed::A", "typed::B", "untyped_status::Traced::", "missing_annotation", "bad"]
        counts = sparse.csr_matrix([[2, 0, 3, 4, 5], [0, 7, 1, 0, 0]])
        pooled, vocabulary, record = _collapse_vocabulary(
            counts, np.array(["one", "one"]), raw,
            strategy="bounded_hierarchical_partner_categories",
            minimum_development_type_count=2, rare_partner_mass_fraction=.5,
            partner_hierarchy={"A": {"subclass": "shared"}, "B": {"subclass": "shared"}},
            canonical_target=True,
        )
        self.assertEqual(record["structurally_empty_pools_removed"], ["other_typed"])
        canonical, names, mapping = canonicalize_counts(
            counts, raw, record["raw_to_pool"], vocabulary
        )
        self.assertEqual(names[-6:], list(CANONICAL_ENDPOINTS))
        np.testing.assert_array_equal(canonical.sum(axis=1), counts.sum(axis=1))
        decoder = CanonicalDecoder.fit(canonical, mapping, len(vocabulary))
        np.testing.assert_array_equal(decoder.pooled_counts(canonical), pooled.toarray())
        self.assertEqual(canonical[:, names.index("out_of_vocabulary")].sum(), 5)
        np.testing.assert_allclose(decoder.decode_probabilities(np.ones(len(vocabulary)) / len(vocabulary)).sum(), 1)

    def test_source_cv_refits_decoder_without_validation_rows(self):
        seed = 127
        permutation = np.random.default_rng(seed).permutation(16)
        fold_ids = (np.arange(16) % 2)[np.argsort(permutation)]
        expected = []
        for fold in range(2):
            training, validation = fold_ids != fold, fold_ids == fold
            mass = self.left[training].sum(axis=0) + .5
            pool_mass = np.bincount(self.mapping, weights=mass)
            rows = self.left[validation]
            correction = (gammaln(self.pooled_left[validation] + 1).sum(axis=1)
                          - gammaln(rows + 1).sum(axis=1)
                          + rows @ np.log(mass / pool_mass[self.mapping])) / rows.sum(axis=1)
            expected.append(correction.mean())
        fake = SimpleNamespace(parameter_count=1, score_neurons=lambda y, x: np.zeros(len(y)))
        with patch("ep12_executor.scientific_models.fit_predictive_model", return_value=fake):
            selection = cross_validated_candidate(
                self.pooled_left, self.covariates, ["T"], seed=seed, folds=2,
                canonical_counts=sparse.csr_matrix(self.left), canonical_to_pool=self.mapping,
            )
        np.testing.assert_allclose(selection.fold_scores, expected, atol=1e-12)
        self.assertAlmostEqual(selection.mean_score, np.mean(expected))
        self.assertAlmostEqual(selection.standard_error, np.std(expected, ddof=1) / np.sqrt(2))

    def test_transfer_scores_use_their_source_decoder_and_canonical_cv(self):
        options = dict(integration_draws=16, folds=2, component_candidates=(2,))
        result = reciprocal_transfer(
            self.pooled_left, self.covariates, self.pooled_right, self.covariates,
            seed=37, left_canonical_counts=self.left, right_canonical_counts=self.right,
            canonical_to_pool=self.mapping, **options,
        )
        fitted = fit_triplet(
            self.pooled_left, self.covariates, seed=seed_from(37, "left-source"),
            canonical_counts=self.left, canonical_to_pool=self.mapping, **options,
        )
        expected = fitted.score_target(self.pooled_right, self.covariates, canonical_counts=self.right)
        np.testing.assert_allclose(result["left_to_right"]["M"], expected["M"], atol=1e-12)
        for family in ("T", "U", "M"):
            pooled_score = getattr(fitted, family).score_neurons(self.pooled_right, self.covariates)
            np.testing.assert_allclose(
                result["left_to_right"][family] - pooled_score,
                self.decoder.correction(self.right, self.pooled_right), atol=1e-12,
            )
        # Reweighting only the held-out target changes its score, not source fit/CV.
        before = fitted.canonical_decoder.log_conditional_probability.copy()
        fitted.score_target(self.pooled_right, self.covariates,
                            canonical_counts=self.right[:, [1, 0, 3, 2, 5, 4]])
        np.testing.assert_array_equal(before, fitted.canonical_decoder.log_conditional_probability)
        self.assertEqual(fitted.reference_model_id,
                         fitted.U.model_id if fitted.U_cv.mean_score > fitted.T_cv.mean_score + 1e-6 else "T")

    def test_singleton_decoder_preserves_existing_generated_scores(self):
        identity = CanonicalDecoder.fit(self.pooled_left, np.arange(3), 3)
        # Equivalent sparse/dense summation orders need only agree at floating
        # precision, far below the scientific score tolerance (1e-6).
        np.testing.assert_allclose(identity.correction(self.pooled_right, self.pooled_right),
                                   np.zeros(16), atol=1e-12, rtol=0)
        with self.assertRaisesRegex(ValueError, "conserve category mass"):
            self.decoder.correction(self.right, self.pooled_right + 1)


class CorrectedReadinessTests(unittest.TestCase):
    def test_old_qualification_cannot_release_corrected_pipeline(self):
        root = EPISODE_ROOT / "outputs"
        with patch("ep12_executor.production_readiness.verify_contract_roles", return_value={}), \
                patch("ep12_executor.production_readiness.verify_full_synthetic_qualification", return_value={}), \
                patch("ep12_executor.production_readiness.verify_observed_provisional_chain", return_value={}), \
                patch("ep12_executor.production_readiness.verify_falsifier_plan", return_value={}), \
                patch("ep12_executor.production_readiness.verify_development_budget_override", return_value={}):
            report = evaluate_production_readiness(
                qualification_dir=root / "generated-fixture", observed_run_root=root / "generated-fixture",
                observed_finalization_dir=root / "generated-fixture",
                falsifier_plan_path=root / "generated-fixture.json",
            )
        self.assertFalse(report["production_readiness_passed"])
        self.assertEqual(report["failed_foundational_checks"], ["canonical_pipeline_qualification"])

    def _check_chain(self, contract_id, protocol):
        plans = [[{"trial_id": f"{stage}_{i}"} for i in range(n)]
                 for stage, n in (("coverage", 18), ("cycle1", 9), ("cycle2", 9))]
        report = {"status": "completed", "data_role": "development_only",
                  "final_connectivity_accessed": False,
                  "scientific_contract_id": contract_id, "score_target_protocol": protocol,
                  "aggregate": {"technical_failure_count": 0}}
        archive = {"verified_valid_trial_count": 36, "final_connectivity_accessed": False,
                   "final_connectivity_access_authorized": False}
        with tempfile.TemporaryDirectory() as name, \
                patch("ep12_executor.production_readiness._configurations", side_effect=plans), \
                patch("ep12_executor.production_readiness.validate_development_configuration"), \
                patch("ep12_executor.production_readiness._read_json", side_effect=[report] * 36 + [archive]):
            root = Path(name)
            return verify_observed_provisional_chain(
                run_root=root, observed_finalization_dir=root / "finalization",
                policy=SimpleNamespace(minimum_valid_trials=36), outputs_root=root,
            )

    def test_v3_contract_roles_do_not_relabel_historical_epochs(self):
        root = EPISODE_ROOT / "outputs/executor"
        result = verify_contract_roles(
            prefix_contract_path=root / "SCIENTIFIC_CONTRACT_POST36_V3.yaml",
            active_contract_path=root / "SCIENTIFIC_CONTRACT_POST36_V3.yaml",
        )
        self.assertEqual(result["contract_roles"]["prefix_epoch"], "observed_prefix_v3")
        self.assertEqual(result["contract_roles"]["active_epoch"], "post36_active_v3")
        old = verify_contract_roles(prefix_contract_path=root / "SCIENTIFIC_CONTRACT.yaml",
                                    active_contract_path=root / "SCIENTIFIC_CONTRACT_POST36.yaml")
        self.assertEqual(old["contract_roles"]["active_epoch"], "post36_active_v2")

    def test_stale_prefix_cannot_qualify_as_corrected_v3(self):
        for contract, protocol in (("ep12_outgoing_tum_v1", None),
                                   ("ep12_outgoing_tum_post36_v2", DECODER_PROTOCOL),
                                   ("ep12_outgoing_tum_post36_v3", None)):
            with self.subTest(contract=contract, protocol=protocol):
                with self.assertRaisesRegex(ReadinessError, "corrected canonical V3 replay"):
                    self._check_chain(contract, protocol)
        result = self._check_chain("ep12_outgoing_tum_post36_v3", DECODER_PROTOCOL)
        self.assertEqual(result["independent_trial_count"], 36)


if __name__ == "__main__":
    unittest.main()
