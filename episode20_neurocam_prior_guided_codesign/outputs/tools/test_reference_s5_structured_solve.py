"""Focused exact-solve and physical propagation checks; Slurm only."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from reference_s5_family import AirFamily, Parameters
from reference_s5_structured_solve import StructuredCapacitanceSolve, StructuredCachedAirFamily
import test_reference_s5_cached_expm as cached_checks


class TestStructuredSolve(unittest.TestCase):
    def test_inverse_transpose_matrix_rhs_and_grounded_boundaries(self):
        rng = np.random.default_rng(820)
        for structure in ("R2", "R3"):
            for rows, cols, probe in ((1, 1, (0, 0)), (3, 4, (1, 1)), (4, 3, (0, 2))):
                for coupling in (0., .2e-12, 30e-12, 3e-9):
                    with self.subTest(structure=structure, shape=(rows, cols), coupling=coupling):
                        family = AirFamily(structure, Parameters(2e-7, coupling),
                            rows=rows, cols=cols, probe=probe, adc_count=2)
                        solve = StructuredCapacitanceSolve(family)
                        for columns in (1, 3):
                            rhs = rng.normal(size=(family.size, columns)) * 1e-12
                            if columns == 1:
                                rhs = rhs[:, 0]
                            saved = rhs.copy()
                            expected = family.cap_factor.solve(rhs)
                            for trans in ("N", "T", "H"):
                                result = solve.solve(rhs, trans=trans)
                                assert_allclose(result, expected, atol=2e-14, rtol=2e-12)
                                residual = family.capacitance @ result - rhs
                                self.assertLess(np.max(np.abs(residual)) / np.max(np.abs(rhs)), 2e-12)
                            np.testing.assert_array_equal(rhs, saved)

    def test_rejects_changed_circuit_and_reduced_family(self):
        family = AirFamily("R1", Parameters(), rows=2, cols=3, probe=(0, 1), adc_count=2)
        with self.assertRaisesRegex(ValueError, "unreduced"):
            StructuredCapacitanceSolve(family)
        family = AirFamily("R2", Parameters(), rows=2, cols=3, probe=(0, 1), adc_count=2)
        family.capacitance[0, 0] *= 1.000001
        with self.assertRaisesRegex(ValueError, "separable"):
            StructuredCapacitanceSolve(family)

    def test_full_augmented_propagation_including_holds(self):
        helper = cached_checks.TestCachedAction()
        for structure in ("R2", "R3"):
            original = helper.family(AirFamily, structure)
            structured = helper.family(StructuredCachedAirFamily, structure)
            for index, (row, slot, marker, start) in enumerate(
                    ((1, 3, .005, 0.), (1, 3, .005, .125), (2, 2, -.003, .25))):
                state = np.sin(np.arange(original.size) + index*.3) * 1e-6
                args = (row, slot, marker, start, start + 2.**-20)
                result = structured.sparse_step(state, *args)
                helper.assert_result(result, original.sparse_step(state, *args))
                helper.assert_result(result, helper.dense_reference(original, state, *args))


if __name__ == "__main__":
    unittest.main()
