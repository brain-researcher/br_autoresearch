"""Targeted equivalence/cache-safety checks; run only in coordinated Slurm."""
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

from reference_s5_family import AirFamily, Parameters, OMEGA
from reference_s5_cached_expm import CachedAirFamily


class TestCachedAction(unittest.TestCase):
    def family(self, cls, structure, **kwargs):
        return cls(structure, Parameters(2e-7, 7e-12), rows=3, cols=4,
                   adc_count=2, probe=(1, 1), cache_bytes=8 * 1024**2, **kwargs)

    def assert_result(self, actual, expected):
        assert_allclose(actual[0], expected[0], atol=2e-12, rtol=2e-8)
        assert_allclose(actual[1], expected[1], atol=2e-20, rtol=2e-8)

    def dense_reference(self, family, state, row, slot, marker, start, end):
        n = family.size
        g, forcing = family.matrices(row, slot)
        a = np.zeros((n + 3 + family.adc_count,) * 2)
        a[:n, :n] = -np.linalg.solve(family.capacitance.toarray(), g.toarray())
        a[:n, n:n+3] = forcing * [1., 1., marker]
        a[n, n+1], a[n+1, n] = OMEGA, -OMEGA
        a[n+3:, :n] = family.hold_observer.toarray()
        initial = np.r_[state, math.sin(OMEGA*start), math.cos(OMEGA*start),
                        1., np.zeros(family.adc_count)]
        result = expm((end-start)*a) @ initial
        return result[:n], result[n+3:]

    def test_full_state_and_integrals_changed_phase_state_and_exact_keys(self):
        # Binary-exact durations let changed-phase cases hit the same matrix
        # plan. A one-ULP duration change MUST then create a separate entry.
        duration = 2. ** -20
        for structure in ("R2", "R3"):
            reference = self.family(AirFamily, structure)
            cached = self.family(CachedAirFamily, structure)
            cases = [(1, 3, .005, 0., duration),
                     (1, 3, .005, .125, .125 + duration),
                     (1, 3, -.003, .125, .125 + duration),
                     (2, 3, -.003, .125, .125 + duration),
                     (2, 2, -.003, .125, .125 + duration),
                     (2, 2, -.003, 0., np.nextafter(duration, np.inf))]
            for i, args in enumerate(cases):
                state = np.sin(np.arange(cached.size) + i*.3) * (i+1)*1e-6
                actual = cached.sparse_step(state, *args)
                self.assert_result(actual, reference.sparse_step(state, *args))
                self.assert_result(actual, self.dense_reference(reference, state, *args))
            self.assertEqual(cached.plan_report()["hits"], 1)
            self.assertEqual(cached.plan_report()["misses"], 5)

    def test_entry_eviction_and_fresh_product_budget_on_hit(self):
        family = self.family(CachedAirFamily, "R2", plan_cache_entries=2)
        state = np.zeros(family.size)
        for slot in (0, 1, 2):
            family.sparse_step(state, 1, slot, .005, 0., 2.**-20)
        self.assertEqual(family.plan_report()["entries"], 2)
        self.assertEqual(family.plan_report()["evictions"], 1)
        with self.assertRaisesRegex(RuntimeError, "operator-product budget"):
            family.sparse_step(state, 1, 2, .005, 0., 2.**-20, max_products=0)
        # The failed interval's counter must not poison the next fresh operator.
        family.sparse_step(state, 1, 2, .005, 0., 2.**-20)
        self.assertEqual(family.plan_report()["hits"], 2)

    def test_rail_certificate_repeated_on_cache_hit(self):
        family = self.family(CachedAirFamily, "R3")
        original = family.certify_controlled_rails
        calls = []
        def checked(state, row, start):
            calls.append(start)
            return original(state, row, start)
        family.certify_controlled_rails = checked
        for start in (0., .125):
            family.sparse_step(np.zeros(family.size), 1, 3, .005, start, start + 2.**-20)
        self.assertEqual(calls, [0., .125])
        self.assertEqual(family.plan_report()["hits"], 1)


if __name__ == "__main__":
    unittest.main()
