from __future__ import annotations

import copy
import unittest
from types import MappingProxyType

import numpy as np
import ep12_executor.fitted_null as fitted_null_module

from ep12_executor.fitted_null import (
    NULL_REPLICATE_COUNT,
    ExpansionRule,
    FittedNullDesign,
    FittedNullError,
    TypePooledExpansionPlan,
    freeze_null_seed_manifest,
    null_seed_binding,
    sample_fitted_null,
    verify_null_seed_manifest,
)
from ep12_executor.scientific_models import FittedPredictiveModel, helmert_ilr_basis


SELECTION_SHA256 = "ab" * 32


def _design() -> FittedNullDesign:
    raw_counts = np.asarray(
        [
            [40, 12, 4, 3, 1],
            [50, 6, 2, 4, 0],
            [45, 9, 3, 2, 1],
            [55, 3, 7, 1, 4],
            [60, 0, 0, 5, 5],
            [52, 0, 0, 4, 4],
            [48, 0, 0, 2, 2],
            [64, 0, 0, 3, 3],
        ],
        dtype=np.int64,
    )
    return FittedNullDesign.create(
        neuron_ids=[101, 102, 103, 104, 201, 202, 203, 204],
        focal_types=["A", "A", "A", "A", "B", "B", "B", "B"],
        sides=["L", "L", "R", "R", "L", "L", "R", "R"],
        covariates=np.asarray(
            [
                [-1.0, 0.0],
                [-0.5, 0.2],
                [0.5, -0.2],
                [1.0, 0.0],
                [-0.8, 0.1],
                [-0.3, -0.1],
                [0.3, 0.1],
                [0.8, -0.1],
            ],
            dtype=float,
        ),
        raw_counts=raw_counts,
        raw_vocabulary=[
            "typed::x",
            "typed::y",
            "typed::z",
            "untyped_status::Traced::Reviewed",
            "missing_annotation",
        ],
        collapsed_vocabulary=[
            "typed::x",
            "other_typed",
            "untyped",
            "missing_annotation",
        ],
        raw_to_collapsed=[0, 1, 1, 2, 3],
        collapsed_typed_bins=["typed::x", "other_typed"],
    )


def _model(
    design: FittedNullDesign,
    model_id: str = "T",
    preferred_typed_bin: str = "other_typed",
) -> FittedPredictiveModel:
    partner_count = len(design.collapsed_vocabulary)
    dimension = partner_count - 1
    basis = helmert_ilr_basis(partner_count)
    coefficients = np.zeros((design.covariates.shape[1] + 1, dimension), dtype=float)
    # Endpoint logits do not matter because the sampler conditions on typed
    # mass.  The extreme finite separation makes row-wise generator selection
    # exactly testable after floating-point softmax underflow.
    if preferred_typed_bin == "typed::x":
        desired_logits = np.asarray([1000.0, -1000.0, -1000.0, -1000.0])
    elif preferred_typed_bin == "other_typed":
        desired_logits = np.asarray([-1000.0, 1000.0, -1000.0, -1000.0])
    else:
        raise ValueError("unknown synthetic preferred typed bin")
    coefficients[0] = desired_logits @ basis
    if model_id == "U_linear":
        prior_means = np.zeros((1, dimension), dtype=float)
        prior_covariances = (np.eye(dimension) * 0.25)[None, :, :]
        prior_weights = np.ones(1, dtype=float)
    else:
        prior_means = np.zeros((1, dimension), dtype=float)
        prior_covariances = (np.eye(dimension) * 1e-10)[None, :, :]
        prior_weights = np.ones(1, dtype=float)
    return FittedPredictiveModel(
        model_id=model_id,
        basis=basis,
        nuisance_coefficients=coefficients,
        latent_offsets=np.zeros((1, dimension), dtype=float),
        log_weights=np.zeros(1, dtype=float),
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        prior_weights=prior_weights,
        integration_draws=1,
        integration_seed=7,
        parameter_count=int(coefficients.size),
        diagnostics={},
        pseudocount=0.5,
    )


def _generators(
    design: FittedNullDesign,
    model_id: str = "T",
    *,
    preferred_by_key: dict[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str], FittedPredictiveModel]:
    keys = sorted(set(zip(design.focal_types, design.sides)))
    preferred_by_key = preferred_by_key or {}
    return {
        key: _model(
            design,
            model_id,
            preferred_by_key.get(key, "other_typed"),
        )
        for key in keys
    }


def _manifest(
    design: FittedNullDesign,
    generators: dict[tuple[str, str], FittedPredictiveModel],
    plan: TypePooledExpansionPlan,
) -> dict[str, object]:
    return freeze_null_seed_manifest(
        master_seed=20260925,
        binding=null_seed_binding(
            design,
            generators,
            plan,
            selection_sha256=SELECTION_SHA256,
        ),
    )


class FittedNullSeedTests(unittest.TestCase):
    def test_seed_manifest_freezes_exactly_99_replicates_and_detects_tampering(self) -> None:
        design = _design()
        generators = _generators(design)
        plan = TypePooledExpansionPlan.build(design)
        binding = null_seed_binding(
            design,
            generators,
            plan,
            selection_sha256=SELECTION_SHA256,
        )
        self.assertEqual(binding["generator_count"], 4)
        self.assertEqual(
            [(row["focal_type"], row["side"]) for row in binding["generators"]],
            [("A", "L"), ("A", "R"), ("B", "L"), ("B", "R")],
        )
        reversed_binding = null_seed_binding(
            design,
            dict(reversed(list(generators.items()))),
            plan,
            selection_sha256=SELECTION_SHA256,
        )
        self.assertEqual(binding, reversed_binding)
        changed_generators = dict(generators)
        changed_generators[("B", "R")] = _model(
            design, "T", preferred_typed_bin="typed::x"
        )
        changed_binding = null_seed_binding(
            design,
            changed_generators,
            plan,
            selection_sha256=SELECTION_SHA256,
        )
        self.assertNotEqual(
            binding["generator_manifest_sha256"],
            changed_binding["generator_manifest_sha256"],
        )
        first = freeze_null_seed_manifest(master_seed=41, binding=binding)
        second = freeze_null_seed_manifest(master_seed=41, binding=binding)
        changed = freeze_null_seed_manifest(master_seed=41, binding=changed_binding)
        self.assertEqual(first, second)
        self.assertNotEqual(first["replicates"][0]["seed"], changed["replicates"][0]["seed"])
        self.assertEqual(first["replicate_count"], NULL_REPLICATE_COUNT)
        seeds = verify_null_seed_manifest(first, expected_binding=binding)
        self.assertEqual(len(seeds), 99)
        self.assertEqual(len(set(seeds)), 99)

        tampered = copy.deepcopy(first)
        tampered["replicates"][0]["seed"] ^= 1
        with self.assertRaises(FittedNullError):
            verify_null_seed_manifest(tampered, expected_binding=binding)
        wrong_binding = dict(binding)
        wrong_binding["selection_sha256"] = "cd" * 32
        with self.assertRaises(FittedNullError):
            verify_null_seed_manifest(first, expected_binding=wrong_binding)


class FittedNullSamplingTests(unittest.TestCase):
    def test_expansion_uses_type_pooled_both_sides_then_development_fallback(self) -> None:
        design = _design()
        plan = TypePooledExpansionPlan.build(design)
        pooled_rule = plan.rules[1]
        # Type A has y:z mass 30:16 after pooling both L and R.
        np.testing.assert_allclose(
            pooled_rule.probabilities_by_type["A"],
            np.asarray([30.0, 16.0]) / 46.0,
        )
        self.assertEqual(
            pooled_rule.source_by_type["A"], "focal_type_pooled_both_sides"
        )
        # Type B has no y/z observations, so it uses the development-wide
        # proportions (which are the same 30:16 here).
        np.testing.assert_allclose(
            pooled_rule.probabilities_by_type["B"],
            np.asarray([30.0, 16.0]) / 46.0,
        )
        self.assertEqual(
            pooled_rule.source_by_type["B"],
            "development_wide_pooled_both_sides_fallback",
        )
        self.assertIn(("B", "other_typed"), plan.fallback_type_bin_pairs)

    def test_T_sample_preserves_rows_covariates_totals_and_exact_endpoint_counts(self) -> None:
        design = _design()
        generators = _generators(design, "T")
        plan = TypePooledExpansionPlan.build(design)
        sample = sample_fitted_null(
            design,
            generators,
            plan,
            seed_manifest=_manifest(design, generators, plan),
            replicate_index=0,
            selection_sha256=SELECTION_SHA256,
        )
        self.assertEqual(sample.neuron_ids, design.neuron_ids)
        self.assertEqual(sample.focal_types, design.focal_types)
        self.assertEqual(sample.sides, design.sides)
        np.testing.assert_array_equal(sample.covariates, design.covariates)
        np.testing.assert_array_equal(
            sample.raw_counts[:, ~design.typed_raw_mask],
            design.raw_counts[:, ~design.typed_raw_mask],
        )
        np.testing.assert_array_equal(
            sample.raw_counts[:, design.typed_raw_mask].sum(axis=1),
            design.raw_counts[:, design.typed_raw_mask].sum(axis=1),
        )
        np.testing.assert_array_equal(
            sample.raw_counts.sum(axis=1), design.raw_counts.sum(axis=1)
        )
        np.testing.assert_array_equal(
            sample.collapsed_counts[:, ~design.collapsed_typed_mask],
            np.asarray(
                [
                    design.raw_counts[:, design.raw_to_collapsed == index].sum(axis=1)
                    for index in np.flatnonzero(~design.collapsed_typed_mask)
                ]
            ).T,
        )
        self.assertGreater(sample.fallback_allocations_with_positive_mass, 0)
        self.assertFalse(sample.raw_counts.flags.writeable)
        self.assertFalse(sample.covariates.flags.writeable)

    def test_T_and_U_samples_replay_exactly_and_replicates_change(self) -> None:
        design = _design()
        plan = TypePooledExpansionPlan.build(design)
        for model_id in ("T", "U_linear"):
            with self.subTest(model_id=model_id):
                generators = _generators(design, model_id)
                manifest = _manifest(design, generators, plan)
                first = sample_fitted_null(
                    design,
                    generators,
                    plan,
                    seed_manifest=manifest,
                    replicate_index=7,
                    selection_sha256=SELECTION_SHA256,
                )
                replay = sample_fitted_null(
                    design,
                    generators,
                    plan,
                    seed_manifest=manifest,
                    replicate_index=7,
                    selection_sha256=SELECTION_SHA256,
                )
                another = sample_fitted_null(
                    design,
                    generators,
                    plan,
                    seed_manifest=manifest,
                    replicate_index=8,
                    selection_sha256=SELECTION_SHA256,
                )
                np.testing.assert_array_equal(first.raw_counts, replay.raw_counts)
                self.assertNotEqual(first.replicate_seed, another.replicate_seed)
                self.assertFalse(np.array_equal(first.raw_counts, another.raw_counts))
                np.testing.assert_array_equal(
                    another.raw_counts.sum(axis=1), design.raw_counts.sum(axis=1)
                )

    def test_selects_the_frozen_generator_for_each_type_and_side(self) -> None:
        design = _design()
        preferences = {
            ("A", "L"): "typed::x",
            ("A", "R"): "other_typed",
            ("B", "L"): "other_typed",
            ("B", "R"): "typed::x",
        }
        generators = {
            key: _model(
                design,
                "U_linear" if key[0] == "B" else "T",
                preference,
            )
            for key, preference in preferences.items()
        }
        sample = sample_fitted_null(
            design,
            generators,
            plan := TypePooledExpansionPlan.build(design),
            seed_manifest=_manifest(design, generators, plan),
            replicate_index=3,
            selection_sha256=SELECTION_SHA256,
        )
        typed_totals = design.raw_counts[:, design.typed_raw_mask].sum(axis=1)
        for row_index, key in enumerate(zip(design.focal_types, design.sides)):
            preferred_index = 0 if preferences[key] == "typed::x" else 1
            other_index = 1 - preferred_index
            self.assertEqual(
                sample.collapsed_counts[row_index, preferred_index],
                typed_totals[row_index],
            )
            self.assertEqual(sample.collapsed_counts[row_index, other_index], 0)
        self.assertEqual(
            sample.generator_models_by_type_side,
            (
                ("A", "L", "T"),
                ("A", "R", "T"),
                ("B", "L", "U_linear"),
                ("B", "R", "U_linear"),
            ),
        )

    def test_missing_and_extra_type_side_generator_keys_fail_closed(self) -> None:
        design = _design()
        complete = _generators(design)
        plan = TypePooledExpansionPlan.build(design)
        missing = dict(complete)
        missing.pop(("B", "R"))
        with self.assertRaises(FittedNullError):
            null_seed_binding(
                design,
                missing,
                plan,
                selection_sha256=SELECTION_SHA256,
            )
        extra = dict(complete)
        extra[("C", "L")] = _model(design)
        with self.assertRaises(FittedNullError):
            null_seed_binding(
                design,
                extra,
                plan,
                selection_sha256=SELECTION_SHA256,
            )

    def test_fail_closed_on_mixed_bins_missing_side_or_unsupported_expansion(self) -> None:
        base = _design()
        with self.assertRaises(FittedNullError):
            FittedNullDesign.create(
                neuron_ids=base.neuron_ids,
                focal_types=base.focal_types,
                sides=base.sides,
                covariates=base.covariates,
                raw_counts=base.raw_counts,
                raw_vocabulary=base.raw_vocabulary,
                collapsed_vocabulary=base.collapsed_vocabulary,
                raw_to_collapsed=[0, 1, 1, 0, 3],
                collapsed_typed_bins=["typed::x", "other_typed"],
            )
        with self.assertRaises(FittedNullError):
            FittedNullDesign.create(
                neuron_ids=base.neuron_ids,
                focal_types=base.focal_types,
                sides=["L", "L", "R", "R", "L", "L", "L", "L"],
                covariates=base.covariates,
                raw_counts=base.raw_counts,
                raw_vocabulary=base.raw_vocabulary,
                collapsed_vocabulary=base.collapsed_vocabulary,
                raw_to_collapsed=base.raw_to_collapsed,
                collapsed_typed_bins=["typed::x", "other_typed"],
            )

        counts_with_empty_raw_bin = np.column_stack(
            [base.raw_counts[:, :3], np.zeros(len(base.neuron_ids), dtype=np.int64), base.raw_counts[:, 3:]]
        )
        unsupported = FittedNullDesign.create(
            neuron_ids=base.neuron_ids,
            focal_types=base.focal_types,
            sides=base.sides,
            covariates=base.covariates,
            raw_counts=counts_with_empty_raw_bin,
            raw_vocabulary=[
                "typed::x",
                "typed::y",
                "typed::z",
                "typed::never_observed",
                "untyped_status::Traced::Reviewed",
                "missing_annotation",
            ],
            collapsed_vocabulary=[
                "typed::x",
                "other_typed",
                "untyped",
                "missing_annotation",
                "typed::never_observed",
            ],
            raw_to_collapsed=[0, 1, 1, 4, 2, 3],
            collapsed_typed_bins=[
                "typed::x",
                "other_typed",
                "typed::never_observed",
            ],
        )
        with self.assertRaises(FittedNullError):
            TypePooledExpansionPlan.build(unsupported)

    def test_rejects_M_as_a_null_generator(self) -> None:
        design = _design()
        generators = _generators(design)
        plan = TypePooledExpansionPlan.build(design)
        generators[("A", "L")] = _model(design, "M2")
        with self.assertRaises(FittedNullError):
            null_seed_binding(
                design,
                generators,
                plan,
                selection_sha256=SELECTION_SHA256,
            )

    def test_expansion_plan_digest_is_bound_and_tampering_fails_closed(self) -> None:
        design = _design()
        generators = _generators(design)
        plan = TypePooledExpansionPlan.build(design)
        manifest = _manifest(design, generators, plan)
        sample = sample_fitted_null(
            design,
            generators,
            plan,
            seed_manifest=manifest,
            replicate_index=0,
            selection_sha256=SELECTION_SHA256,
        )
        self.assertEqual(sample.expansion_plan_sha256, plan.plan_sha256)
        self.assertEqual(
            manifest["binding"]["expansion_plan_sha256"], plan.plan_sha256
        )

        original = plan.rules[1]
        altered_rules = dict(plan.rules)
        altered_probabilities = dict(original.probabilities_by_type)
        altered_probabilities["A"] = (1.0, 0.0)
        altered_rules[1] = ExpansionRule(
            collapsed_index=original.collapsed_index,
            raw_indices=original.raw_indices,
            development_probabilities=original.development_probabilities,
            probabilities_by_type=MappingProxyType(altered_probabilities),
            source_by_type=original.source_by_type,
        )
        tampered = TypePooledExpansionPlan(
            design_sha256=plan.design_sha256,
            rules=MappingProxyType(altered_rules),
            fallback_type_bin_pairs=plan.fallback_type_bin_pairs,
            plan_sha256=plan.plan_sha256,
        )
        with self.assertRaisesRegex(FittedNullError, "expansion plan digest mismatch"):
            sample_fitted_null(
                design,
                generators,
                tampered,
                seed_manifest=manifest,
                replicate_index=0,
                selection_sha256=SELECTION_SHA256,
            )

        rehashed = TypePooledExpansionPlan(
            design_sha256=tampered.design_sha256,
            rules=tampered.rules,
            fallback_type_bin_pairs=tampered.fallback_type_bin_pairs,
            plan_sha256=fitted_null_module._digest_json(
                fitted_null_module._expansion_plan_payload_parts(
                    tampered.design_sha256,
                    tampered.rules,
                    tampered.fallback_type_bin_pairs,
                )
            ),
        )
        with self.assertRaisesRegex(
            FittedNullError, "differs from deterministic type-pooled expansion"
        ):
            sample_fitted_null(
                design,
                generators,
                rehashed,
                seed_manifest=manifest,
                replicate_index=0,
                selection_sha256=SELECTION_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
