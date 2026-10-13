"""Small independent arithmetic and information-boundary fixtures."""
import copy
import itertools
import unittest

import numpy as np

from calibrate_fixed_hw_correction import AMPLITUDES, calibrated_prediction, score_amplitudes, select_amplitude
from primary_reference import ANIMALS, SESSIONS


def score_banks():
    return {session: [{"amplitude": a, "delta_R2": 0., "identity_increment": 0.}
                      for a in AMPLITUDES] for session in SESSIONS}


class CalibrationTests(unittest.TestCase):
    def test_literal_endpoints_and_single_baseline(self):
        b = np.full((16, 15, 2), 7.)
        hw = b+4
        self.assertIs(calibrated_prediction(b, hw, 0.), b)
        self.assertIs(calibrated_prediction(b, hw, 1.), hw)
        np.testing.assert_array_equal(calibrated_prediction(b, hw, .25), b+1)
        with self.assertRaises(ValueError):
            calibrated_prediction(b, hw, .6)

    def test_held_outcome_and_identity_cannot_choose_amplitude(self):
        banks = score_banks()
        for rows in banks.values():
            rows[2]["delta_R2"] = 1.
        held = SESSIONS[0]
        before = select_amplitude(banks, held)
        banks[held][4]["delta_R2"] = 1e9
        for rows in banks.values():
            rows[0]["identity_increment"] = 1e12
        after = select_amplitude(banks, held)
        self.assertEqual(before, after)
        self.assertNotIn(held, after[1])
        self.assertEqual(len(after[1]), 15)

    def test_equal_animal_not_equal_session_weighting(self):
        banks = score_banks()
        for session, rows in banks.items():
            rows[1]["delta_R2"] = 1. if ANIMALS[session] == "Mihili" else 0.
            rows[2]["delta_R2"] = .4
        index, _, scores = select_amplitude(banks, SESSIONS[0])
        self.assertEqual(index, 1)
        self.assertEqual(scores[1], .5)
        self.assertAlmostEqual(scores[2], .4)

    def test_exact_ties_preserve_larger_amplitude(self):
        banks = score_banks()
        self.assertEqual(select_amplitude(banks, SESSIONS[0])[0], 4)
        for rows in banks.values():
            rows[1]["delta_R2"] = rows[3]["delta_R2"] = 1.
        self.assertEqual(select_amplitude(banks, SESSIONS[0])[0], 3)
        incomplete = copy.deepcopy(banks)
        incomplete.pop(SESSIONS[-1])
        with self.assertRaises(ValueError):
            select_amplitude(incomplete, SESSIONS[0])

    def test_primary_and_derangement_are_independent_direct_sums(self):
        rng = np.random.default_rng(9081)
        labels = np.repeat(np.arange(8), 3)
        b = np.repeat(rng.normal(size=(8, 15, 2)), 3, axis=0)
        y = b+rng.normal(size=b.shape)
        hw = b+.8*(y-b)+rng.normal(size=b.shape)
        denominator = np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum()
        rows = score_amplitudes(y, b, hw, labels)
        for row in rows:
            a = row["amplitude"]
            p = b+a*(hw-b)
            correct = np.square(y-p).sum()
            wrong = 0.
            for direction in range(8):
                yd, pd = y[labels == direction], p[labels == direction]
                losses = [np.square(yd-pd[list(order)]).sum()
                          for order in itertools.permutations(range(3))
                          if all(i != donor for i, donor in enumerate(order))]
                wrong += sum(losses)/len(losses)
            self.assertAlmostEqual(row["delta_R2"], (np.square(y-b).sum()-correct)/denominator, places=12)
            self.assertAlmostEqual(row["identity_increment"], (wrong-correct)/denominator, places=12)
            self.assertAlmostEqual(row["identity_increment"], a*rows[-1]["identity_increment"], places=12)


if __name__ == "__main__":
    unittest.main()
