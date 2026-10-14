"""Two focused source-fixture checks; coordinator-owned compute only.

Reuse the qualified sequence bridge fixture. No empirical artifacts, model
search, recovery simulation or EEG inference is run by these checks.
"""
from dataclasses import asdict, replace
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from continuous_fold_runner import FIXED_PANEL_MODELS, SequencePredictors
from continuous_sequence_model import JointSequenceDesign
from continuous_tuning import _verify_base
from prepare_continuous_eeg import prepare_participant_sequences
from production_continuous_designs import (BASELINE_TIE, CAPACITY_TIE, NUISANCE_RECIPE,
    FALSE_PARTITION_RECIPE, SPAN, TemporalBasis, TrainingDesignBuilder, _kernels,
    _require_candidate_context_recipe, make_tuning_recipe, validate_recipe)
from slot_capacity import RIDGE_GRID
from test_prepare_continuous_eeg import fixture, raw_reader
from training_transforms import CORE_IMAGE, LABEL


def recipe():
    # Explicit invented test recipe: this is not the empirical selection file.
    return dict(recipe_id="fixture_only", image_panel="core_image",
        image_ranks={key: 32 for key in CORE_IMAGE}, label_ranks={key: 32 for key in LABEL},
        temporal_basis="binned_FIR_20ms", label_kernel_bandwidth=1., nuisance_recipe=NUISANCE_RECIPE,
        intercept_penalty=0., nuisance_penalty=1e-6, shared_feature_penalty=1e-3,
        fit_weight_scale="mean_sequences", fit_weight_multiplier=1., normalization="ratio_of_equal_sequence_means",
        covariance_centered=True, covariance_denominator="sum_weights", covariance_target="trace_over_61_identity",
        shrinkage_strength=.1, capacity_tie_break=CAPACITY_TIE, baseline_tie_break=BASELINE_TIE,
        base_rank_relative_tolerance=1e-12, rbf_numerics=dict(eigenvalue_rtol=1e-12,
            eigenvalue_atol=0., reconstruction_rtol=1e-10, reconstruction_max_atol=1e-9),
        solver=dict(atol=1e-8, btol=1e-8, conlim=1e12, maxiter=10000),
        metadata_byte_cap=16*1024**2, npz_byte_cap=16*1024**2,
        working_array_byte_cap=16*1024**3, dense_design_byte_limit=32*1024**3)


class ProductionDesignTests(unittest.TestCase):
    def test_full_panel_shared_bases_coarse_flags_and_actual_tick_convolution(self):
        arguments = fixture()
        sequences = prepare_participant_sequences(**arguments, read_samples=raw_reader)
        sequence = sequences[0]
        source = arguments["main_source_events"]
        events = {
            "sample_zero_based": np.asarray([int(row["sample"]) - 1 for row in source]),
            "concept": np.asarray([int(row["objectnumber"]) for row in source]),
            "block": np.asarray([int(row["blocksequencenumber"]) for row in source]),
            "within_block_sequence": np.asarray([int(row["withinsequencenumber"]) for row in source]),
            "target_flag": np.asarray([int(i % 7 == 0) for i in range(len(source))]),
            "response_flag": np.asarray([int(i % 11 == 0) for i in range(len(source))])}
        indices = np.arange(309)
        builder = TrainingDesignBuilder(participant_id="fixture", events=MappingProxyType(events),
            false_groups=np.asarray([(events["concept"] + i + 1) % 1854 for i in range(8)]),
            predictor_metadata=MappingProxyType({sequence.sequence_id:
                (sequence.preparation_input.span, sequence.sample_times_seconds)}),
            split=None, recipe=recipe(), image_transform=None, label_transform=None,
            label_kernel=SimpleNamespace(training_features=np.arange(1854*4).reshape(1854, 4) / 1000.),
            image_features=np.arange(len(source)*3).reshape(len(source), 3) / 1000.,
            label_features=np.arange(1854*2).reshape(1854, 2) / 1000.,
            rows_by_sequence={sequence.sequence_id: indices}, basis=TemporalBasis("binned_FIR_20ms"), check=lambda: None)
        view = SequencePredictors(sequence.sequence_id, sequence.block, sequence.within_block,
                                  sequence.sample_times_seconds, sequence.preparation_input)
        designs = builder(view)
        self.assertEqual(set(designs), set(FIXED_PANEL_MODELS))
        self.assertEqual(designs["nuisance"].shape[1], 284)
        self.assertEqual(designs["reduced_task_nuisance"].shape[1], 204)
        np.testing.assert_array_equal(designs["nuisance"].nuisance[:, :204],
                                      designs["reduced_task_nuisance"].nuisance)
        for name in FIXED_PANEL_MODELS:
            if name != "reduced_task_nuisance":
                np.testing.assert_array_equal(designs[name].nuisance, designs["nuisance"].nuisance)
        _verify_base(SimpleNamespace(sequences=(designs["image"],)),
                     SimpleNamespace(sequences=(designs["label_linear"],)))
        for name in ("label_capacity_matched", "true_concept", *(f"false_group_{i}" for i in range(1, 9))):
            _verify_base(SimpleNamespace(sequences=(designs["label_linear"],)),
                         SimpleNamespace(sequences=(designs[name],)))
            self.assertIs(designs[name].terms["image"].kernels, designs["image"].terms["image"].kernels)
        true = designs["true_concept"]
        beta = np.zeros(true.shape[1])
        beta[true.terms["true_concept"].column_slice.start + int(events["concept"][0]) * 40 + 1] = 1.
        ticks = np.rint(sequence.sample_times_seconds * 1000).astype(int)
        lag = ticks - 5  # First fixture onset is exactly source-relative 5 ms.
        np.testing.assert_array_equal(JointSequenceDesign([true]) @ beta, ((lag >= 20) & (lag < 40)).astype(float))
        expected_target = np.zeros((len(ticks), 40))
        for event_index in range(309):
            if events["target_flag"][event_index]:
                lags = ticks - (event_index + 5)
                valid = (lags >= 0) & (lags <= 800)
                expected_target[np.flatnonzero(valid), np.minimum(lags[valid] // 20, 39)] += 1
        np.testing.assert_array_equal(designs["nuisance"].nuisance[:, 204:244], expected_target)
        self.assertFalse(true.terms["true_concept"].features.flags.writeable)
        self.assertFalse(true.terms["image"].kernels[0][2].flags.writeable)
        with self.assertRaises(ValueError):
            builder(replace(view, preparation_input=replace(sequence.preparation_input,
                source_sample_indices=sequence.preparation_input.source_sample_indices + 1)))
        with self.assertRaises(ValueError):
            builder(replace(view, sample_times_seconds=view.sample_times_seconds + .001))

    def test_explicit_recipe_basis_edges_training_covariance_and_deterministic_ties(self):
        context = dict(recipe_id=FALSE_PARTITION_RECIPE, recorded_span_recipe=asdict(SPAN),
                       filter_left_support_samples=128, filter_right_support_samples=0)
        _require_candidate_context_recipe(context)
        changes = (dict(recorded_span_recipe=dict(asdict(SPAN), post_last_inclusive_samples=799)),
                   dict(filter_left_support_samples=127), dict(filter_right_support_samples=1),
                   dict(recipe_id="unselected_context_recipe"), dict(filter_left_support_samples=True),
                   dict(filter_right_support_samples=False))
        for change in changes:
            with self.subTest(context_change=tuple(change)), self.assertRaises(ValueError):
                _require_candidate_context_recipe(dict(context, **change))
        selected = recipe()
        incomplete = dict(selected)
        incomplete.pop("shared_feature_penalty")
        with self.assertRaises(ValueError):
            validate_recipe(incomplete)
        fir = TemporalBasis("binned_FIR_20ms")
        np.testing.assert_array_equal(np.argmax(fir(np.asarray([0., .02, .799, .8])), axis=1), [0, 1, 39, 39])
        lo, hi, kernels = _kernels(np.asarray([-1, 0, 20, 800, 801]), (0,), fir)[0]
        self.assertEqual((lo, hi), (1, 4))
        np.testing.assert_array_equal(kernels.sum(axis=1), np.ones(3))
        for width in (8, 12, 16):
            basis = TemporalBasis(f"raised_cosine_{width}")
            values = basis(np.arange(801) / 1000.)
            # Subtracting represented centers and dividing by their spacing
            # amplifies floating-point coordinate error by O(width). Adjacent
            # raised cosines sum to one analytically; retain the actual basis
            # values and allow coordinate/cosine/addition roundoff only.
            np.testing.assert_allclose(values.sum(axis=1), 1., rtol=0.,
                                       atol=8*width*np.finfo(np.float64).eps)
            self.assertEqual(values[0, 0], 1.)
            self.assertEqual(values[-1, -1], 1.)
        tuning = make_tuning_recipe(selected)
        residual = np.asarray([np.arange(61), np.arange(61) + 2.], dtype=float)
        choices = tuning.covariance(residual, np.asarray([.25, .75]))
        self.assertTrue(choices.center_covariance)
        self.assertEqual(choices.covariance_denominator, 1.)
        np.testing.assert_allclose(choices.shrinkage_target, np.eye(61) * .75)
        choice = tuning.select_baseline(SimpleNamespace(feasible_concept_indices=(0, 1),
            matched_label_mean_losses={0: 1., 1: 1.}, image_mean_losses=(1.,) * 13))
        self.assertEqual((choice.concept_index, choice.image_index), (1, 12))
        curves = {"true_concept": (10.,) * 13}
        curves.update({name: (10.,) * 13 for name in ("label_capacity_matched", *(f"false_group_{i}" for i in range(1, 9)))})
        matched = tuning.match_capacity(SimpleNamespace(feasible=SimpleNamespace(concept_index=2,
            label_indices=(0, 1), false_indices=((0, 1),) * 8),
            capacity=SimpleNamespace(curve=lambda name: SimpleNamespace(term_edf=curves[name]))))
        self.assertEqual((matched.concept_index, matched.label_index, matched.false_indices), (2, 1, (1,) * 8))
        vector = tuning.image_concept_penalties(SimpleNamespace(nuisance_penalties=(0., 1e-6),
            selected_image_index=3, selected_concept_index=4, image_columns=5, full_columns=9))
        self.assertEqual(vector, (0., 1e-6) + (RIDGE_GRID[3],) * 3 + (RIDGE_GRID[4],) * 4)


if __name__ == "__main__":
    unittest.main()
