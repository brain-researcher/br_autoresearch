"""Bit-exactness of fast_io_exact against legacy scalar code and math.fsum."""
import math
import unittest

import numpy as np

from fast_io_exact import (bitwise_equal, float_bits, fsum_rows, last_window, midpoint, shift,
                           upper_div, upper_mul, upper_result)
from raw_input_hold import _midpoint
from raw_world_time import _upper_div, _upper_mul, _upper_result
from repeating_acquisition_schedule import _last_window, _shift


def _bits(value):
    return float_bits(np.array([value]))[0]


class FsumTests(unittest.TestCase):
    def assert_rows(self, rows, counts=None):
        rows = np.asarray(rows, dtype=np.float64)
        counts = np.full(len(rows), rows.shape[1]) if counts is None else np.asarray(counts)
        expected = np.array([math.fsum(row[:count].tolist()) for row, count in zip(rows, counts)])
        actual = fsum_rows(rows, counts)
        mismatch = np.flatnonzero(float_bits(expected) != float_bits(actual))
        self.assertEqual(len(mismatch), 0, f"first mismatches: {[(rows[i].tolist(), expected[i], actual[i]) for i in mismatch[:5]]}")

    def test_random_wide_exponents_and_widths(self):
        rng = np.random.Generator(np.random.PCG64(1))
        for width in (1, 2, 3, 4, 6):
            mantissa = rng.uniform(-1, 1, (200000, width))
            exponent = rng.integers(-60, 60, (200000, width)).astype(float)
            rows = mantissa * 2.0 ** exponent
            counts = rng.integers(0, width + 1, 200000)
            self.assert_rows(rows, counts)

    def test_cancellation_ties_and_signed_zero(self):
        eps = np.finfo(float).eps
        cases = [[1e-16, 1., 1e16], [1., -1., 0.], [-0., -0., -0.], [0., -0.], [-0.],
                 [1., eps / 2, eps / 2 ** 54], [1., eps / 2, -eps / 2 ** 54], [2. ** 53, 1., 1.],
                 [2. ** 53, 1., -2. ** -60], [1e308, 1e308 * 0.5, -1e308], [5e-324, 5e-324, -5e-324],
                 [3., -2. ** 53, 2. ** 53], [0.1, 0.2, 0.3], [-0.1, 0.1, -0.]]
        width = max(map(len, cases))
        rows = np.zeros((len(cases), width))
        counts = np.array([len(case) for case in cases])
        for i, case in enumerate(cases):
            rows[i, :len(case)] = case
        self.assert_rows(rows, counts)

    def test_quantizer_level_triples_and_near_cancellation(self):
        step = 0.02 / 2 ** 18
        rng = np.random.Generator(np.random.PCG64(2))
        index = rng.integers(0, 2 ** 18, (300000, 3))
        levels = -0.01 + (index + 0.5) * step
        self.assert_rows(levels)
        mirrored = np.stack((levels[:, 0], -levels[:, 0], levels[:, 2]), axis=1)
        self.assert_rows(mirrored)
        perturbed = np.stack((levels[:, 0], np.nextafter(-levels[:, 0], 0), levels[:, 1] * 2 ** -40), axis=1)
        self.assert_rows(perturbed)


class ScalarReplicaTests(unittest.TestCase):
    def test_shift_matches_legacy(self):
        rng = np.random.Generator(np.random.PCG64(3))
        cycles = np.concatenate(([0, 1, 2, 3998, 3999], rng.integers(0, 4000, 300)))
        relative = np.concatenate(([0.0, -0.0, 0.004, 5e-7, 2.5e-6, 0.0039975],
                                   np.arange(0, 8001, 7) / 2000000, rng.uniform(0, 0.004, 200)))
        actual = shift(cycles[:, None], relative[None, :])
        for i, cycle in enumerate(cycles.tolist()):
            expected = np.array([_shift(cycle, value) for value in relative.tolist()])
            self.assertTrue(bitwise_equal(actual[i], expected), cycle)

    def test_last_window_matches_legacy(self):
        ends = (np.arange(4000) + 1) * 0.004
        rng = np.random.Generator(np.random.PCG64(4))
        times = np.concatenate((ends, np.nextafter(ends, 0), np.nextafter(ends, 1), [0.0, 1e-9, 16.0, 17.0],
                                rng.uniform(0, 16, 5000)))
        expected = [_last_window(value) for value in times.tolist()]
        self.assertEqual(last_window(times).tolist(), expected)

    def test_outward_operations_and_midpoint_match_legacy(self):
        rng = np.random.Generator(np.random.PCG64(5))
        left = np.concatenate(([0.0, 5e-324, 1.0], rng.uniform(0, 1, 2000) * 10.0 ** rng.integers(-30, 10, 2000)))
        right = np.concatenate(([3.0, 0.0, 7.0], rng.uniform(1e-3, 1, 2000) * 10.0 ** rng.integers(-30, 10, 2000)))
        self.assertTrue(bitwise_equal(upper_mul(left, right), [_upper_mul(a, b) for a, b in zip(left, right)]))
        self.assertTrue(bitwise_equal(upper_div(left, right[3:].tolist() + [1.0, 2.0, 3.0]),
                                      [_upper_div(a, b) for a, b in zip(left, right[3:].tolist() + [1.0, 2.0, 3.0])]))
        self.assertTrue(bitwise_equal(upper_result(left), [_upper_result(a) for a in left]))
        starts = np.concatenate(([0.0, 0.004 * 3999], rng.uniform(0, 15.99, 3000)))
        ends = starts + np.concatenate(([2.5e-6, 2.5e-6], rng.uniform(1e-8, 1e-5, 3000)))
        middle, radius = midpoint(starts, ends)
        expected = [_midpoint(a, b) for a, b in zip(starts.tolist(), ends.tolist())]
        self.assertTrue(bitwise_equal(middle, [m for m, _ in expected]))
        self.assertTrue(bitwise_equal(radius, [r for _, r in expected]))


if __name__ == "__main__":
    unittest.main()
