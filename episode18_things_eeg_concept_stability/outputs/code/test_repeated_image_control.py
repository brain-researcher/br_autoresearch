"""Focused mathematical/leakage checks; execute only in coordinated Slurm."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import repeated_image_control as model
from continuous_sequence_model import JointSequenceDesign, SequenceDesign
from core_reference_direct_ridge import DirectCoreRidge
from repeated_image_sequences import RepeatedParticipant, RepeatedSequence
from training_transforms import CommonSensorSpace


def recipe():
    path = Path(__file__).resolve().parents[1] / "config/core_recipe_20261013_001.json"
    return json.loads(path.read_text())


def sequence(ordinal):
    times = np.arange(1000) / 250.
    onsets = .8 + np.arange(200) * .012
    # Known array fixture only, never a simulated scientific population.
    values = np.sin((np.arange(len(times))[:, None] + 1) * (np.arange(62)[None, :] + 1) * .037 + ordinal)
    return RepeatedSequence("s%02d" % ordinal, ordinal, times, values,
        tuple("c%02d" % i for i in range(62)), np.ones(len(times), dtype=bool),
        onsets, np.roll(np.arange(200), ordinal), np.arange(200) % 7 == 0,
        np.arange(200) % 13 == 0, 0., float(onsets[-1] + .05))


def tiny_sequences():
    """Small algebra fixture for the fold machinery, not the 200-column adapter."""
    result, designs = [], {}
    times = np.arange(24) / 25.
    for ordinal in range(12):
        samples = np.sin((np.arange(24)[:, None] + 1) * (np.arange(62)[None, :] + 1) * .17 + ordinal * .21)
        item = SimpleNamespace(sequence_id="s%02d" % ordinal, ordinal=ordinal,
            sensor_samples=samples, channel_order=tuple("c%02d" % i for i in range(62)),
            retained_mask=np.arange(24) % 5 != 0)
        nuisance = np.column_stack((np.ones(24), times))
        features = np.column_stack((nuisance, np.sin(times * 5 + ordinal * .1), np.cos(times * 9)))
        designs[item.sequence_id] = {
            "nuisance": SequenceDesign(item.sequence_id, times, nuisance, {}),
            "identity": SequenceDesign(item.sequence_id, times, features, {}),
        }
        result.append(item)
    return tuple(result), designs


class RepeatedControlTests(unittest.TestCase):
    def test_full_identity_control_has_exact_nuisance_and_actual_event_support(self):
        item = sequence(0)
        designs = model._sequence_designs(item)
        nuisance, identity = designs["nuisance"], designs["identity"]
        self.assertEqual(nuisance.shape[1], 88)
        self.assertEqual(identity.shape[1], 2488)
        np.testing.assert_array_equal(nuisance.nuisance, identity.nuisance)
        term = identity.terms["identity"]
        np.testing.assert_array_equal(term.features, np.eye(200)[item.image_indices])
        self.assertEqual(identity.schema[1][0][2:4], (12, (0., .8)))
        self.assertEqual(tuple(term.column_slice.indices(identity.shape[1]))[:2], (88, 2488))
        basis = model.TemporalBasis("raised_cosine_12")
        ticks = np.rint(item.sample_times_seconds * 1000).astype(int)
        expected_cue = basis((ticks - round(item.cue_seconds * 1000)) / 1000.)
        expected_offset = basis((ticks - round(item.final_offset_seconds * 1000)) / 1000.)
        np.testing.assert_allclose(nuisance.nuisance[:, 40:52], expected_cue, atol=1e-13)
        np.testing.assert_allclose(nuisance.nuisance[:, 52:64], expected_offset, atol=1e-13)
        np.testing.assert_allclose(nuisance.nuisance[:, 1],
            2 * item.sample_times_seconds / (item.event_onsets_seconds[-1] + .801) - 1)
        # One identity's event response must neither wrap nor appear at other identities.
        beta = np.zeros(identity.shape[1]); beta[88 + 17 * 12] = 1
        predicted = JointSequenceDesign([identity]) @ beta
        expected = basis((ticks - round(item.event_onsets_seconds[17] * 1000)) / 1000.)[:, 0]
        np.testing.assert_allclose(predicted, expected, atol=1e-13)

    def test_identity_and_nuisance_keep_exact_zero_and_800ms_endpoints(self):
        original = sequence(0)
        onsets = .2 + np.arange(200) * .02
        times = np.arange(1251) / 250.
        item = replace(original, sample_times_seconds=times, event_onsets_seconds=onsets,
                       final_offset_seconds=float(onsets[-1] + .05))
        designs = model._sequence_designs(item)
        identity, nuisance = designs["identity"], designs["nuisance"]
        # Late1.2s and1.4s onsets end at2.0s and2.2s. In particular, binary
        # 2.2-1.4 exceeds binary0.8; floating absolute subtraction loses a sample.
        for image, start, stop in ((50, 300, 501), (60, 350, 551)):
            lo, hi, kernel = identity.terms["identity"].kernels[image]
            self.assertEqual((lo, hi), (start, stop))
            self.assertEqual((kernel[0, 0], kernel[-1, -1]), (1., 1.))
            for basis_index, endpoint in ((0, start), (11, stop - 1)):
                beta = np.zeros(identity.shape[1]); beta[88 + image * 12 + basis_index] = 1
                prediction = JointSequenceDesign([identity]) @ beta
                self.assertEqual(prediction[endpoint], 1.)
                self.assertEqual((prediction[start - 1], prediction[stop]), (0., 0.))
        for basis_index in (0, 11):
            # Summing that basis across all200 identities equals the original
            # nuisance event-intercept expansion at every sample, endpoints too.
            beta = np.zeros(identity.shape[1]); beta[88 + basis_index::12] = 1.
            np.testing.assert_array_equal(JointSequenceDesign([identity]) @ beta,
                                          nuisance.nuisance[:, 4 + basis_index])

    def test_complete_chronological_roster_and_ratio_of_means(self):
        person = RepeatedParticipant("sub-test", tuple("i%03d" % i for i in range(200)),
            tuple(sequence(i) for i in reversed(range(12))), {})
        ordered = model._roster(person, lambda: None)
        self.assertEqual([item.ordinal for item in ordered if item.ordinal % 2 == 0], [0, 2, 4, 6, 8, 10])
        self.assertEqual([item.ordinal for item in ordered if item.ordinal % 2 == 1], [1, 3, 5, 7, 9, 11])
        self.assertAlmostEqual(float(1 - model._normal([4., 1.], [1., 9.])), .5)
        with self.assertRaises(ValueError):
            model._normal([1., 1.], [0., 0.])
        with self.assertRaises(ValueError):
            model._roster(replace(person, sequences=person.sequences[:-1]), lambda: None)
        invalid = replace(person.sequences[0], image_indices=np.zeros(200, dtype=int))
        with self.assertRaises(ValueError):
            model._roster(replace(person, sequences=(invalid,) + person.sequences[1:]), lambda: None)

    def test_outer_EEG_does_not_enter_selection_or_whitening(self):
        sequences, designs = tiny_sequences()
        training = sequences[::2]
        heldout = sequences[1::2]
        sensor = CommonSensorSpace(training[0].channel_order)
        outcomes, seen_training = [], []
        original_prepare = model._prepare_training
        def observed_prepare(items, *args):
            seen_training.append(tuple(item.sequence_id for item in items))
            return original_prepare(items, *args)
        for changed in (False, True):
            test = heldout
            if changed:
                test = tuple(SimpleNamespace(**{**item.__dict__,
                    "sensor_samples": item.sensor_samples * 100 + np.arange(62)[None, :]}) for item in heldout)
            with tempfile.TemporaryDirectory() as directory, patch.object(model, "_prepare_training", side_effect=observed_prepare):
                outcomes.append(model._fit_direction(training, test, direction="odd_to_even",
                    designs=designs, sensor=sensor, recipe=recipe(), directory=Path(directory),
                    binding={"test": "same_training"}, backend=DirectCoreRidge(check=lambda: None, chunk_rows=13),
                    check=lambda: None, progress=lambda stage: None))
        first, second = outcomes
        self.assertEqual(first["selected_penalty_index"], second["selected_penalty_index"])
        np.testing.assert_array_equal(first["inner_mean_normalized_identity_losses"], second["inner_mean_normalized_identity_losses"])
        self.assertEqual(first["training_whitening"], second["training_whitening"])
        self.assertNotEqual(first["identity_sequence_losses"], second["identity_sequence_losses"])
        allowed = {item.sequence_id for item in training}
        self.assertEqual([len(ids) for ids in seen_training], [5] * 6 + [6] + [5] * 6 + [6])
        self.assertTrue(all(set(ids).issubset(allowed) for ids in seen_training))
        self.assertTrue(all(set(ids).isdisjoint(item.sequence_id for item in heldout) for ids in seen_training))

    def test_incomplete_or_altered_checkpoint_is_never_reused(self):
        binding = {"recipe": "fixed"}
        record = {"status": "complete_inner", "binding": binding,
            "training_sequence_ids": ["a", "b"], "heldout_sequence_ids": ["c"],
            "identity_sequence_losses": [[2.]] * 13, "nuisance_sequence_losses": [4.],
            "normalized_identity_losses": [.5] * 13}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inner.json"
            model._json_write(path, record)
            self.assertEqual(model._load_inner(path, binding, ("a", "b"), "c"), record)
            for mutation in ({"status": "incomplete_inner"}, {"identity_sequence_losses": [[2.]] * 12},
                             {"normalized_identity_losses": [.4] * 13}, {"training_sequence_ids": ["a", "c"]}):
                model._json_write(path, {**record, **mutation})
                with self.assertRaises(ValueError):
                    model._load_inner(path, binding, ("a", "b"), "c")

    def test_second_direction_failure_never_publishes_partial_person(self):
        person = RepeatedParticipant("sub-test", tuple("i%03d" % i for i in range(200)),
            tuple(sequence(i) for i in range(12)), {})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(model, "_sequence_designs", return_value={}), patch.object(model, "_fit_direction",
                    side_effect=[{"direction": "odd_to_even", "gain": .2}, RuntimeError("job interrupted")]):
                with self.assertRaises(RuntimeError):
                    model.run_participant(person, recipe=recipe(), run_dir=directory,
                        budget_check=lambda: None, progress=lambda stage: None)
            self.assertFalse((Path(directory) / "complete.json").exists())
            self.assertTrue((Path(directory) / "odd_to_even.complete.json").exists())

    def test_original_numerical_tolerances_and_larger_lambda_tie_remain_fixed(self):
        original = recipe()
        self.assertEqual(model.validate_recipe(original), original)
        altered = deepcopy(original); altered["solver"]["atol"] = 1e-6
        with self.assertRaises(ValueError):
            model.validate_recipe(altered)
        record = {"status": "complete_direction", "binding": {}, "direction": "odd_to_even",
            "training_sequence_ids": list("abcdef"), "heldout_sequence_ids": list("ghijkl"),
            "identity_sequence_losses": [1.] * 6, "nuisance_sequence_losses": [2.] * 6,
            "inner_mean_normalized_identity_losses": [1.] * 13, "inner_validations_completed": 6,
            "selected_penalty_index": 12, "selected_ridge_penalty": 1e6, "gain": .5,
            "training_whitening": {"sequence_ids": list("abcdef"), "shrinkage_strength": .1,
                                   "regularized_covariance_trace": 1.}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direction.json"
            model._json_write(path, record)
            model._load_direction(path, {}, "odd_to_even", tuple("abcdef"), tuple("ghijkl"))
            record["selected_penalty_index"] = 0; record["selected_ridge_penalty"] = 1e-6
            model._json_write(path, record)
            with self.assertRaises(ValueError):
                model._load_direction(path, {}, "odd_to_even", tuple("abcdef"), tuple("ghijkl"))


if __name__ == "__main__":
    unittest.main()
