"""Focused new-air-operator checks. Run only in the coordinated compute shell."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from reference_s5_family import (AirFamily, Parameters, _SineIntegralOperator,
                                 AMPLITUDE_VOLTS, OMEGA)
from reference_s5_operator import (AirRecord, SineSummary, FRAMES, nearest_summary, ratio_db)


class TestAirFamily(unittest.TestCase):
    def family(self, structure):
        return AirFamily(structure, Parameters(2e-7, 7e-12), rows=3, cols=4,
                         adc_count=2, probe=(1, 1), cache_bytes=8 * 1024**2)

    def test_dirichlet_derivative_and_sparse_adjoint(self):
        for structure in ("R1", "R2", "R3"):
            f = self.family(structure); n = f.size
            z = np.sin(np.arange(n) + .7) * 1e-5
            row, slot, time, marker, duration = 1, 2, .1234567, .003, 2e-7
            full_g, marker_current = f.full_conductance(row, slot)
            g, forcing = f.matrices(row, slot)
            x = f.reconstruct(z, time)
            derivative = AMPLITUDE_VOLTS * OMEGA * math.cos(OMEGA * time)
            expected_rhs = np.asarray(f.projection.T @ (-full_g @ x + marker_current * marker)).ravel()
            expected_rhs -= f.cap_probe * derivative
            observed = -g @ z + f.capacitance @ (forcing @ np.array([math.sin(OMEGA*time), math.cos(OMEGA*time), marker]))
            assert_allclose(observed, expected_rhs, atol=2e-23, rtol=3e-12)
            self.assertGreater(np.linalg.norm(f.cap_probe * derivative), 0.)
            op = _SineIntegralOperator(f, g, forcing, marker, duration, 10000)
            size = n + 3 + f.adc_count
            direct = np.zeros((size, size))
            direct[:n, :n] = -np.linalg.solve(f.capacitance.toarray(), g.toarray())
            direct[:n, n:n+3] = forcing * np.array([1., 1., marker])
            direct[n, n+1], direct[n+1, n] = OMEGA, -OMEGA
            direct[n+3:, :n] = f.hold_observer.toarray()
            assert_allclose(op @ np.eye(size), duration * direct, atol=1e-12, rtol=3e-12)
            assert_allclose(op.T @ np.eye(size), duration * direct.T, atol=1e-12, rtol=3e-12)

    def test_projection_charge_row_equivalence_and_full_energy(self):
        f = self.family("R1")
        q = np.cos(np.arange(f.full_size)) * 1e-17
        q[f.driven] = 0.
        delta = f.charge_delta(q)
        assert_allclose(f.capacitance @ delta, f.projection.T @ q, atol=1e-30, rtol=3e-12)
        for slot in range(4):
            left, _ = f.full_conductance(0, slot)
            right, _ = f.full_conductance(2, slot)
            assert_allclose((f.projection.T @ left @ f.projection).toarray(),
                            (f.projection.T @ right @ f.projection).toarray(), atol=1e-20, rtol=3e-12)
        z = np.cos(np.arange(f.size)) * 1e-4; time = .019
        x = f.reconstruct(z, time)
        self.assertAlmostEqual(f.physical_energy(z, time) / (.5 * x @ (f.full_capacitance @ x)), 1., places=13)

    def test_transient_aperture_against_independent_ode(self):
        for structure in ("R1", "R2", "R3"):
            f = self.family(structure); n = f.size
            row, slot, marker, start, end = 1, 1, .005, .123456, .1234562
            z = np.sin(np.arange(n)) * 1e-6
            full_g, marker_current = f.full_conductance(row, slot)
            inv_c = np.linalg.inv(f.capacitance.toarray())
            def rhs(t, value):
                physical = f.reconstruct(value[:n], t)
                current = np.asarray(f.projection.T @ (-full_g @ physical + marker_current * marker)).ravel()
                current -= f.cap_probe * AMPLITUDE_VOLTS * OMEGA * math.cos(OMEGA * t)
                return np.concatenate((inv_c @ current, f.hold_observer @ value[:n]))
            exact = solve_ivp(rhs, (start, end), np.concatenate((z, np.zeros(f.adc_count))),
                              method="DOP853", rtol=2e-12, atol=1e-20).y[:, -1]
            actual, integral = f.sparse_step(z, row, slot, marker, start, end)
            assert_allclose(actual, exact[:n], atol=2e-12, rtol=2e-8)
            assert_allclose(integral, exact[n:], atol=2e-20, rtol=2e-8)
            if structure == "R1":
                endpoint, aperture = f.dense_step(row, slot, marker, end-start)
                initial = np.r_[z, math.sin(OMEGA*start), math.cos(OMEGA*start), 1.]
                assert_allclose((endpoint @ initial)[:n], exact[:n], atol=2e-12, rtol=2e-8)
                assert_allclose(aperture @ initial, exact[n:], atol=2e-20, rtol=2e-8)


class TestS5ObservationAndRestart(unittest.TestCase):
    def test_actual_timestamp_fit_prefix_guard_and_zero_ratio(self):
        s = SineSummary(2)
        with self.assertRaises(ValueError):
            s.amplitudes([0])
        for frame in range(FRAMES):
            times = frame * .004 + np.array([.0001375, .0028175])
            values = np.array([.003, .00002]) * np.sin(OMEGA*times + np.array([.31, -.8])) + [.01, -.001]
            s.add(times, values)
        assert_allclose(s.amplitudes([0, 1]), [.003, .00002], rtol=2e-11, atol=1e-15)
        self.assertEqual(ratio_db(0., 1.)["status"], "negative_infinity_zero_numerator")
        self.assertEqual(ratio_db(1., 0.)["status"], "undefined_zero_denominator")
        # Row±1 is the source mask, column±1 the gate mask; far sites never queried.
        class Spy:
            def amplitudes(self, sites):
                self.sites = sites
                return np.array([1., .1, .2, .003, .002])
        spy = Spy(); near = nearest_summary(spy)
        self.assertEqual(spy.sites, [2015, 1951, 2079, 2014, 2016])
        self.assertAlmostEqual(near["nearest_source"]["db"], 20*math.log10(.2))
        self.assertAlmostEqual(near["nearest_gate"]["db"], 20*math.log10(.003))

    def test_frame_checkpoint_carries_physics_summary_and_uncommitted_rows(self):
        # Small physical circuit plus writer-sized synthetic readout. This tests
        # the restart seam without propagating a second full-array fixture.
        def operator():
            family = AirFamily("R1", Parameters(), rows=2, cols=2, adc_count=1, probe=(0, 0))
            template = SimpleNamespace(digest="focused_test_clock", conversions=3,
                obs_assigned=np.arange(4096) * 1e-8,
                conv_aperture_start=np.array([1e-7, 2e-7, 3e-7]),
                conv_aperture_end=np.array([2e-7, 3e-7, 4e-7]))
            return SimpleNamespace(family=family, template=template, pixel_rows=np.arange(4096))
        def advance(op, state, index):
            q = np.zeros(op.family.full_size); q[op.family.sources[0]] = 3e-17
            state = state + op.family.charge_delta(q)
            state, _ = op.family.sparse_step(state, 0, index % 4, .005,
                                            index*.004, index*.004 + 2e-7)
            value = np.full(4096, state[0]) + np.arange(4096)*1e-10
            result = SimpleNamespace(index=index, aperture_mean=np.array([1., 2., 3.])*state[0],
                code=np.array([1, 2, 3]), raw_clipped=np.zeros(3, bool),
                value=value, clipped=np.zeros(4096, bool),
                assigned=index*.004 + op.template.obs_assigned)
            return state, result, 7e-19
        with TemporaryDirectory() as temp:
            op = operator(); record = AirRecord(Path(temp)/"record", op, {"test": True})
            first = advance(op, record.state, 0); record.commit(*first)
            expected = advance(op, record.state, 1)
            expected_xtx, expected_xty = record.summary.xtx.copy(), record.summary.xty.copy()
            record.file["committed"][1] = True  # durable output ahead of canonical checkpoint
            record.close()
            fresh = operator(); restored = AirRecord(Path(temp)/"record", fresh, {"test": True}, resume=True)
            self.assertEqual(restored.next_frame, 1)
            self.assertFalse(restored.file["committed"][1])
            assert_allclose(restored.summary.xtx, expected_xtx, atol=0, rtol=0)
            assert_allclose(restored.summary.xty, expected_xty, atol=0, rtol=0)
            repeated = advance(fresh, restored.state, 1)
            assert_allclose(repeated[0], expected[0], atol=1e-12, rtol=2e-8)
            assert_allclose(repeated[1].value, expected[1].value, atol=1e-12, rtol=2e-8)
            restored.commit(*repeated)
            self.assertEqual(restored.summary.count, 2)
            self.assertEqual(restored.boundary_energy, 14e-19)
            restored.close()


if __name__ == "__main__":
    unittest.main()
