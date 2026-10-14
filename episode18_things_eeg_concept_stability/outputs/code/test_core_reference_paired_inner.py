"""Two deterministic original-versus-paired checks; coordinator compute only.

Reuse the existing 72-sequence/61-contrast fixture and independent dense ridge
oracle. Only iterative fitting is replaced; preparation, weights, whitening,
held-out scoring, normalization, grid and fold selection execute for real.
No empirical input or new scientific simulation is used.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from continuous_sequence_model import EventTerm, SequenceDesign
import core_reference_paired_inner as paired
import production_core_reference as core
import test_continuous_tuning as prior
from test_production_core_reference import recipe
from training_transforms import CommonSensorSpace


def person_fixture(calls, sequences=None):
    def prepare(training, *, split, budget_check):
        if {row.block for row in training} != set(split.training_blocks):
            raise AssertionError("Only the exact original training blocks may prepare predictors")
        calls.append((split, tuple(row.sequence_id for row in training)))
        # Reuse the original core fixture's 88/64 nuisance and 12-wide event
        # layout. Include a training-only feature center to exercise sharing.
        center = float(np.mean([row.preparation_input % 7 for row in training]))
        def basis(lags):
            values = np.zeros((len(lags), 12))
            values[:, 0] = 1.
            return values
        def build(view):
            if hasattr(view, "sensor_samples"):
                raise AssertionError("Held-out EEG reached a predictor builder")
            nuisance = np.zeros((len(view.sample_times_seconds), 88))
            nuisance[:, 0] = 1.
            term = EventTerm(np.array([0.]), np.array([[1+view.preparation_input % 7-center]]),
                             basis, 12, (0., .8))
            return {"nuisance": SequenceDesign(view.sequence_id, view.sample_times_seconds, nuisance, {}),
                "reduced_task_nuisance": SequenceDesign(view.sequence_id, view.sample_times_seconds, nuisance[:, :64], {}),
                "image": SequenceDesign(view.sequence_id, view.sample_times_seconds, nuisance, {"image":term})}
        return tuning.TuningPreparation(tuple(row.sequence_id for row in training), build)
    return SimpleNamespace(participant_id="fixture", sequences=prior.fixture() if sequences is None else sequences,
        inputs=SimpleNamespace(channel_order=prior.CHANNELS), prepare_training=prepare)


class PairedInnerReferenceTests(unittest.TestCase):
    def test_complete_original_equivalence_coverage_order_and_numerical_fit_reuse(self):
        records = []
        for implementation in (core.fit_person, paired.fit_person):
            prepared, inner, outer = [], [], []
            progress = dict(inner_splits_completed=0, outer_splits_completed=0)
            with patch.object(folds, "_fit", side_effect=prior.dense_fit) as fitted:
                result = implementation(person_fixture(prepared), recipe(), check=lambda:None,
                    retain_inner=inner.append, retain_outer=outer.append, progress=progress)
            records.append((result, prepared, inner, outer, progress, fitted.call_count))
        original, reused = records
        self.assertEqual((original[5], reused[5]), (438, 228))
        self.assertEqual((len(original[1]), len(reused[1])), (36, 21))
        self.assertEqual((len(reused[2]), len(reused[3])), (30, 6))
        self.assertEqual(reused[4]["actual_multivariate_fits"], 228)
        self.assertEqual(reused[4]["original_logical_multivariate_fits"], 438)
        self.assertEqual(reused[4]["unique_inner_training_sets_completed"], 15)
        self.assertEqual((reused[4]["inner_splits_completed"], reused[4]["outer_splits_completed"]), (30,6))
        self.assertEqual(len({ids for split,ids in reused[1] if split.inner_index is not None}), 15)
        before = {(row.split.outer_index,row.split.inner_index):row for row in original[2]}
        after = {(row.split.outer_index,row.split.inner_index):row for row in reused[2]}
        self.assertEqual(set(after), {(i,j) for i in range(6) for j in range(6) if i != j})
        for key, row in after.items():
            expected = before[key]
            self.assertEqual(row.split, expected.split)
            self.assertEqual(row.training_sequence_ids, expected.training_sequence_ids)
            self.assertEqual(row.heldout_sequence_ids, expected.heldout_sequence_ids)
            self.assertFalse(set(row.training_sequence_ids) & set(row.heldout_sequence_ids))
            np.testing.assert_array_equal(row.image_sequence_losses, expected.image_sequence_losses)
            np.testing.assert_array_equal(row.nuisance_sequence_losses, expected.nuisance_sequence_losses)
            np.testing.assert_array_equal(row.normalized_image_losses, expected.normalized_image_losses)
        for index, (expected, actual) in enumerate(zip(original[3], reused[3])):
            ordered = [after[(index,j)].normalized_image_losses for j in range(6) if j != index]
            mean = np.mean(ordered, axis=0)
            self.assertEqual(actual.result.image_index,
                min(range(13), key=lambda j:(mean[j],-tuning.RIDGE_GRID[j],j)))
            self.assertEqual(actual.result.image_index, expected.result.image_index)
            self.assertEqual(actual.preparation.training_sequence_ids, expected.preparation.training_sequence_ids)
            np.testing.assert_array_equal(actual.whitener.empirical_covariance, expected.whitener.empirical_covariance)
            for name in core.CORE_MODELS:
                np.testing.assert_array_equal(actual.fits[name].coefficients, expected.fits[name].coefficients)
                np.testing.assert_array_equal(actual.penalties[name], expected.penalties[name])
                self.assertEqual(actual.fits[name].stop_codes, expected.fits[name].stop_codes)
        self.assertEqual(reused[0], original[0])

    def test_counterpart_uses_one_training_state_and_separate_heldout_EEG_and_denominators(self):
        selected = recipe()
        sensor = CommonSensorSpace(prior.CHANNELS)
        limits = folds.SolverLimits(**selected["solver"])
        convention = SimpleNamespace(normalization=selected["normalization"])
        split = folds._split(0,1)
        outputs, solutions = [], []
        for change_other_validation in (False, True):
            raw = prior.fixture()
            if change_other_validation:
                changed = []
                for row in raw:
                    values = row.sensor_samples.copy()
                    if row.block in folds.BLOCK_PAIRS[0]:
                        values[:, 0] += 3.  # A retained sensor contrast, not common-average offset.
                    changed.append(replace(row, sensor_samples=values))
                raw = tuple(changed)
            calls, betas = [], []
            person = person_fixture(calls, raw)
            sequences = folds._roster(person.sequences, sensor, lambda:None)
            def fit(*args):
                answer = prior.dense_fit(*args)
                betas.append(answer.coefficients.copy())
                return answer
            with patch.object(folds, "_fit", side_effect=fit) as fitted:
                cache = core._prepare_core(sequences, split, sensor, selected, person.prepare_training,
                    limits, lambda:None, lambda value:None)
                reverse = paired._counterpart(cache, sequences, sensor, convention, lambda:None)
                for field in ("training","preparation","train_designs","targets","weights","penalties",
                              "nuisance_fit","whitener"):
                    self.assertIs(getattr(reverse,field), getattr(cache,field))
                self.assertEqual(reverse.split, folds._split(1,0))
                self.assertEqual(len(calls), 1)
                self.assertFalse(set(row.sequence_id for row in cache.heldout)
                                 & set(row.sequence_id for row in reverse.heldout))
                result = paired._inner_pair(cache, reverse, convention, limits, lambda:None)
            self.assertEqual(fitted.call_count, 14)
            for row in result:
                expected = np.mean(row.image_sequence_losses,axis=1)/np.mean(row.nuisance_sequence_losses)
                np.testing.assert_array_equal(row.normalized_image_losses, expected)
            with self.assertRaisesRegex(ValueError,"one_identical_training_state_required"):
                paired._inner_pair(cache, replace(reverse, weights=reverse.weights.copy()),
                                   convention, limits, lambda:None)
            outputs.append(result)
            solutions.append(betas)
        for first, second in zip(*solutions):
            np.testing.assert_array_equal(first, second)
        # Changing only pair0 must not change the shared fit or pair1 errors.
        np.testing.assert_array_equal(outputs[0][0].image_sequence_losses, outputs[1][0].image_sequence_losses)
        np.testing.assert_array_equal(outputs[0][0].nuisance_sequence_losses, outputs[1][0].nuisance_sequence_losses)
        self.assertFalse(np.array_equal(outputs[0][1].image_sequence_losses, outputs[1][1].image_sequence_losses))
        self.assertFalse(np.array_equal(outputs[0][1].nuisance_sequence_losses, outputs[1][1].nuisance_sequence_losses))


if __name__ == "__main__":
    unittest.main()
