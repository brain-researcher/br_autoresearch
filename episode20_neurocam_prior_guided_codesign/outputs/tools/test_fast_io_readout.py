"""Quantizer and digital-combination kernels against the legacy readout."""
import math
import unittest

import numpy as np

from acquisition_readout import QuantizerTransfer
from conditional_acquisition_recipe import _quantizer
from fast_io_exact import float_bits, fsum_rows
from fast_io_readout import QuantizerTable


class QuantizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transfer = _quantizer(lambda: None)
        cls.table = QuantizerTable.from_transfer(cls.transfer)

    def assert_matches(self, values):
        values = np.asarray(values, dtype=np.float64)
        code, decoded, clipped = self.table.convert(values)
        expected = [self.transfer.convert(value) for value in values.tolist()]
        self.assertEqual(code.tolist(), [item[0] for item in expected])
        self.assertTrue(np.array_equal(float_bits(decoded), float_bits(np.array([item[1] for item in expected]))))
        self.assertEqual(clipped.tolist(), [item[2] for item in expected])

    def test_every_threshold_tie_and_neighbours(self):
        thresholds = self.table.thresholds
        self.assert_matches(thresholds)
        self.assert_matches(np.nextafter(thresholds, np.inf))
        self.assert_matches(np.nextafter(thresholds, -np.inf))

    def test_levels_range_ends_zeros_and_random(self):
        rng = np.random.Generator(np.random.PCG64(6))
        lo, hi = self.table.input_min_volts, self.table.input_max_volts
        special = [0.0, -0.0, lo, hi, np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf), -1.0, 1.0,
                   5e-324, -5e-324, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf)]
        self.assert_matches(special)
        self.assert_matches(self.table.levels[::7])
        self.assert_matches(rng.uniform(-0.0125, 0.0125, 200000))
        self.assert_matches(rng.normal(0, 30e-6, 200000))

    def test_tables_are_the_legacy_payload(self):
        self.assertEqual(tuple(self.table.thresholds.tolist()), self.transfer.thresholds_volts)
        self.assertEqual(tuple(self.table.levels.tolist()), self.transfer.code_volts)
        self.assertNotIn(0.0, self.transfer.code_volts)

    def test_nonfinite_refuses(self):
        with self.assertRaises(ValueError):
            self.table.convert([0.0, math.nan])


class CombinationTests(unittest.TestCase):
    def test_pixel_and_marker_arithmetic_matches_math_fsum(self):
        transfer = _quantizer(lambda: None)
        levels = np.array(transfer.code_volts)
        rng = np.random.Generator(np.random.PCG64(7))
        codes = rng.integers(0, len(levels), (100000, 3))
        values = levels[codes]
        weights = np.ones(3)
        expected = [math.fsum(w * v for w, v in zip(weights.tolist(), row)) / 3.0 for row in values.tolist()]
        actual = fsum_rows(weights * values) / 3.0
        self.assertTrue(np.array_equal(float_bits(actual), float_bits(np.array(expected))))
        single = [math.fsum([1.0 * v]) / 1.0 for v in values[:, 0].tolist()]
        self.assertTrue(np.array_equal(float_bits(fsum_rows(values[:, :1] * 1.0) / 1.0), float_bits(np.array(single))))


if __name__ == "__main__":
    unittest.main()
