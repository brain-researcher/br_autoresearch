"""Two integration checks against existing dense/matcher/diagnostic oracles.

No empirical data, search, recovery campaign or execution allocation is used
by the source fixture. Actual test execution belongs to the coordinator.
"""
from dataclasses import fields, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
import selected_design_diagnostics as diagnostics
from slot_capacity import slot_capacity_curves
import streamed_slot_capacity as backend
import streamed_tuning_adapter as adapter
import test_continuous_tuning as tuning_fixture
import test_selected_design_diagnostics as prior
from test_streamed_slot_capacity import impulse


FINGERPRINTS = tuple(f"{index:064x}" for index in range(8))
RESOURCES = adapter.StreamedResources(2, 2, 16*1024**2)


def fixture(outer=0):
    base, added, weights, penalties = prior.fixture()
    designs = {name: [] for name in folds.FIXED_PANEL_MODELS}
    slot_arrays = {name: (added[:, ::-1] if name in tuning.FALSE_MODELS else added)
                   for name in tuning.MATCHED_MODELS}
    for part, rows in enumerate((slice(0, 4), slice(4, 8))):
        times = np.arange(4, dtype=float)/250
        nuisance = base[rows, :1]
        image = EventTerm(times, base[rows, 1:2], impulse, 1, (0., 0.))
        label = EventTerm(times, base[rows, 2:], impulse, 1, (0., 0.))
        for name in designs:
            terms = {}
            if name not in ("nuisance", "reduced_task_nuisance"):
                terms["image"] = image
            if name in ("label_linear",) + tuning.MATCHED_MODELS:
                terms["label"] = label
            if name in tuning.MATCHED_MODELS:
                terms["slot"] = EventTerm(times, slot_arrays[name][rows], impulse, 1, (0., 0.))
            if name == "image_plus_true_concept":
                terms["slot"] = EventTerm(times, added[rows], impulse, 1, (0., 0.))
            designs[name].append(SequenceDesign(str(part), times, nuisance, terms))
    cache = SimpleNamespace(split=folds._split(outer),
        train_designs={name: JointSequenceDesign(values) for name, values in designs.items()},
        weights=weights, penalties={"label_linear": penalties})
    return cache, tuning_fixture.recipe([], []), base, slot_arrays


def assert_distribution(test, actual, expected):
    for item in fields(expected):
        first, second = getattr(actual, item.name), getattr(expected, item.name)
        if second is None or isinstance(second, (bool, int)):
            test.assertEqual(first, second, item.name)
        else:
            np.testing.assert_allclose(first, second, atol=1e-9, rtol=1e-9, err_msg=item.name)


class StreamedAdapterTests(unittest.TestCase):
    def test_original_all_grid_match_and_distributions_complete_six_folds(self):
        cache, recipe, base, slots = fixture()
        dense = slot_capacity_curves(base_design=base, slot_designs=slots,
            sample_weights=cache.weights, base_penalties=cache.penalties["label_linear"],
            base_rank_relative_tolerance=recipe.base_rank_relative_tolerance)
        inner = adapter.capacity_for_split(cache, recipe, lambda: None, resources=RESOURCES)
        self.assertEqual(tuning.all_feasible_tuples(inner), tuning.all_feasible_tuples(dense))
        self.assertEqual(tuple(curve.slot_name for curve in inner.curves), tuning.MATCHED_MODELS)
        for actual in inner.curves:
            expected = dense.curve(actual.slot_name)
            for name in ("schur_singular_values", "term_edf", "incremental_model_edf", "total_model_edf"):
                np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-9, rtol=1e-9)
        common = diagnostics.prepare_selected_base(recipe_id=diagnostics.RECIPE_ID, base_design=base,
            sample_weights=cache.weights, base_penalties=cache.penalties["label_linear"],
            base_rank_relative_tolerance=1e-12, array_byte_cap=16*1024**2, projection_chunk_rows=2)
        chosen = tuning.BaselineChoice(6, 3)
        reference = diagnostics.evaluate_selected_slot(common, slot_name="true_concept",
            slot_design=slots["true_concept"], selected_penalty_index=6)
        false = {name: diagnostics.evaluate_selected_slot(common, slot_name=name,
            slot_design=slots[name], selected_penalty_index=6) for name in tuning.FALSE_MODELS}
        results = []
        with patch.object(adapter, "streamed_slot_capacity_curves", wraps=backend.streamed_slot_capacity_curves) as numerical:
            for outer in range(6):
                cache.split = folds._split(outer)
                actual = adapter.outer_capacity_and_match(cache, chosen, recipe, lambda: None,
                    resources=RESOURCES, diagnostic_recipe_id=diagnostics.RECIPE_ID,
                    frozen_partition_fingerprints=FINGERPRINTS)
                expected_match = tuning._match(recipe, cache.split, dense,
                    next(value for value in tuning.all_feasible_tuples(dense) if value.concept_index == 6))
                self.assertEqual(actual.match, expected_match)
                expected = diagnostics.compare_selected_outer_fold(outer_index=outer,
                    true_slot=reference, false_slots=false, frozen_partition_fingerprints=FINGERPRINTS)
                self.assertEqual(actual.selected_design.passed, expected.passed)
                for observed, original in zip(actual.selected_design.false_matches, expected.false_matches):
                    self.assertEqual((observed.false_slot, observed.reference_penalty_index,
                                      observed.comparison_penalty_index, observed.term_edf_pass, observed.passed),
                                     (original.false_slot, original.reference_penalty_index,
                                      original.comparison_penalty_index, original.term_edf_pass, original.passed))
                    np.testing.assert_allclose([observed.reference_term_edf, observed.comparison_term_edf,
                                               observed.term_edf_relative_difference],
                                              [original.reference_term_edf, original.comparison_term_edf,
                                               original.term_edf_relative_difference], atol=1e-9, rtol=1e-9)
                    assert_distribution(self, observed.spectrum, original.spectrum)
                    assert_distribution(self, observed.joint_leverage, original.joint_leverage)
                for name, original in (("true_concept", reference), *false.items()):
                    np.testing.assert_allclose(actual.design_singular_values[name], original.design_singular_values,
                                               atol=1e-10, rtol=1e-10)
                    np.testing.assert_allclose(actual.conditional_gram_eigenvalues[name],
                                               original.conditional_gram_eigenvalues, atol=1e-9, rtol=1e-9)
                requests = numerical.call_args.kwargs["leverage_penalty_indices"]
                self.assertEqual(sum(map(len, requests.values())), 105)
                self.assertEqual(requests["true_concept"], (6,))
                self.assertEqual(requests["label_capacity_matched"], ())
                self.assertTrue(all(requests[name] == tuple(range(13)) for name in tuning.FALSE_MODELS))
                results.append(actual)
            self.assertEqual(numerical.call_count, 6)
        assembled = adapter.assemble_outer_checks(results)
        self.assertEqual(assembled["status"], "selected_design_checks_pass")
        self.assertFalse(assembled["EEG_scored"])
        self.assertFalse(assembled["false_bank_fully_qualified"])
        with self.assertRaises(ValueError):
            adapter.assemble_outer_checks(results[:-1])
        with self.assertRaises(ValueError):
            adapter.assemble_outer_checks(results[:-1] + [results[0]])
        foreign = replace(results[-1], selected_design=replace(results[-1].selected_design,
                                                              frozen_partition_fingerprints=FINGERPRINTS[::-1]))
        with self.assertRaises(ValueError):
            adapter.assemble_outer_checks(results[:-1] + [foreign])

    def test_invalid_scope_and_admission_refuse_before_any_factorization(self):
        cache, recipe, _, _ = fixture()
        def outer(selected_cache=cache, **changes):
            arguments = dict(resources=RESOURCES, diagnostic_recipe_id=diagnostics.RECIPE_ID,
                             frozen_partition_fingerprints=FINGERPRINTS)
            return adapter.outer_capacity_and_match(selected_cache, tuning.BaselineChoice(6, 3),
                recipe, lambda: None, **dict(arguments, **changes))
        with patch.object(backend.lapack, "dtpqrt", side_effect=AssertionError("must refuse before QR")):
            with self.assertRaises(MemoryError):
                outer(resources=adapter.StreamedResources(2, 2, 1))
            with self.assertRaises(ValueError):
                outer(frozen_partition_fingerprints=(FINGERPRINTS[0],)*8)
            with self.assertRaises(ValueError):
                outer(diagnostic_recipe_id="unselected")
            with self.assertRaises(ValueError):
                outer(SimpleNamespace(**dict(vars(cache), split=folds._split(0, 1))))
            with self.assertRaises(ValueError):
                outer(SimpleNamespace(**dict(vars(cache), weights=np.zeros(8))))
            incomplete = dict(cache.train_designs)
            incomplete.pop("image_plus_true_concept")
            with self.assertRaises(ValueError):
                outer(SimpleNamespace(**dict(vars(cache), train_designs=incomplete)))
            wrong_width = dict(cache.train_designs)
            wrong_width["false_group_8"] = cache.train_designs["label_linear"]
            with self.assertRaises(ValueError):
                outer(SimpleNamespace(**dict(vars(cache), train_designs=wrong_width)))


if __name__ == "__main__":
    unittest.main()
