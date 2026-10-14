"""Small mathematical checks for dwell accounting; run in Slurm only."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

from fast_solver_engine import JMAX, ModalBlock
from fast_solver_fastpath import prefix_work
from fast_solver_multirate import MultirateSolver
from fast_solver_continuous_kernel import BoundedWeightCache, ContinuousMultirateSolver


class AccountingTests(unittest.TestCase):
    def test_independent_modal_audit_includes_cross_terms_and_off_block_heat(self):
        c = np.array([1.2, 0.7])
        g = np.array([[2.0, -0.4], [-0.4, 3.0]])
        omega, start, duration = np.array([1.7, 2.9]), 0.13, 0.2
        forcing = np.array([[0.3 + 0.2j, -0.1 + 0.25j], [0.08 - 0.11j, 0.17 + 0.04j]])
        particular = np.stack([np.linalg.solve(g + 1j * w * np.diag(c), f)
                               for w, f in zip(omega, forcing)])
        initial_hom = np.array([0.2, -0.1])
        unit = ModalBlock([[c[0]]], [[g[0, 0]]], "audit_unit")
        active = ModalBlock([[c[1]]], [[g[1, 1]]], "audit_active")
        base = SimpleNamespace(model=SimpleNamespace(P=1, S=1), unit_blocks=[unit],
                               unit_members=[np.array([0])])
        audit_modal = {
            "active": (active.to_modal(particular[:, 1:]), forcing[:, 1:] @ active.phi),
            "units": {0: (unit.to_modal(particular[:, :1].reshape(2, 1, 1)),
                           forcing[:, :1].reshape(2, 1, 1) @ unit.phi)}}
        entry = {"block": active, "port_pos": np.array([0]), "port_global": np.array([1]),
                 "g_off_edges": (np.array([0, 1]), np.array([1, 0]), np.array([-0.4, -0.4]))}
        record = {"a": 0.0, "h": duration, "ya0": active.to_modal(initial_hom[1:]),
                  "a_coupling": np.zeros((1, JMAX + 1)),
                  "unit_coupling": np.zeros((1, JMAX + 1)),
                  "unit_y0": [(0, np.array([0]), unit.to_modal(initial_hom[:1].reshape(1, 1)))]}

        def state(t):
            return (np.real(np.sum(particular * np.exp(1j * omega * t)[:, None], axis=0))
                    + initial_hom * np.exp(-np.diag(g) / c * (t - start)))

        def external(t):
            return np.real(np.sum(forcing * np.exp(1j * omega * t)[:, None], axis=0))

        expected_input = quad(lambda t: float(state(t) @ external(t)),
                              start, start + duration, epsabs=1e-14, epsrel=1e-12)[0]
        expected_heat = quad(lambda t: float(state(t) @ g @ state(t)), start, start + duration,
                             epsabs=1e-14, epsrel=1e-12)[0]
        # The constructed path has uncoupled homogeneous modes. Its physical
        # balance is deliberately nonzero: the routine must not force closure.
        e0 = 0.5 * np.sum(c * state(start) ** 2)
        e1 = 0.5 * np.sum(c * state(start + duration) ** 2)
        self.assertGreater(abs(e1 - e0 - expected_input + expected_heat), 1e-7)
        for implementation in (MultirateSolver, ContinuousMultirateSolver):
            solver = implementation.__new__(implementation)
            solver.base = base
            solver.row = solver._audit_row = 0
            solver._audit_rules = {}
            solver.dwell = duration
            solver.tones = SimpleNamespace(omega=omega)
            solver._audit_modal = audit_modal
            orders = (16, 32)
            if implementation is ContinuousMultirateSolver:
                solver.weight_cache = BoundedWeightCache(1024**2)
                solver._audit_weights_cache = solver.weight_cache.scope("audit")
                solver.stats = dict(audit_weight_cache_hits=0, audit_weight_cache_misses=0)
                orders += orders  # Check the independent oracle on cache hits too.
            for order in orders:
                with self.subTest(implementation=implementation.__name__, order=order):
                    actual = solver.upstream_energy_audit(entry, record, start, order)
                    self.assertAlmostEqual(actual["input_work"], expected_input, delta=1e-13)
                    self.assertAlmostEqual(actual["heat"], expected_heat, delta=1e-13)
                    self.assertGreater(abs(e1 - e0 - actual["input_work"] + actual["heat"]), 1e-7)
            if implementation is ContinuousMultirateSolver:
                self.assertEqual(solver.stats, dict(audit_weight_cache_hits=4, audit_weight_cache_misses=4))

    def test_independent_audit_with_multimode_member_polynomial_forcing(self):
        # Two members share a two-mode unit block; their polynomial drives
        # differ. The independent trajectory below uses physical coordinates.
        c = np.array([1.2, 0.7, 1.2, 0.7, 0.9])
        unit_g = np.array([[2., -0.4], [-0.4, 3.]])
        local_g = np.zeros((5, 5))
        local_g[:2, :2] = local_g[2:4, 2:4] = unit_g
        local_g[4, 4] = 2.3
        g = local_g.copy()
        g[0, 4] = g[4, 0] = -0.13
        g[2, 4] = g[4, 2] = -0.19
        omega, start, duration = np.array([1.7, 2.9]), 0.13, 0.2
        forcing = np.array([[0.3 + 0.2j, -0.1 + 0.25j, 0.12 - 0.08j, 0.04 + 0.06j, -0.2 + 0.1j],
                            [0.08 - 0.11j, 0.17 + 0.04j, -0.07 + 0.03j, 0.09 - 0.02j, 0.13 + 0.05j]])
        particular = np.stack([np.linalg.solve(g + 1j * w * np.diag(c), f)
                               for w, f in zip(omega, forcing)])
        initial_hom = np.array([0.2, -0.1, -0.12, 0.17, 0.08])
        unit_coupling = np.zeros((2, JMAX + 1))
        unit_coupling[:, :3] = [[0.3, -0.2, 0.07], [-0.25, 0.13, -0.11]]
        active_coupling = np.zeros((1, JMAX + 1))
        active_coupling[0, :3] = [0.11, 0.05, -0.04]

        # Augment x'=-C^-1 G_local x+C^-1 drive*q with q=(1,s,s^2),
        # s=(t-start)/duration. This oracle uses no modal response weights.
        drive = np.zeros((5, 3))
        drive[[0, 2]] = unit_coupling[:, :3]
        drive[4] = active_coupling[0, :3]
        generator = np.zeros((8, 8))
        generator[:5, :5] = -local_g / c[:, None]
        generator[:5, 5:] = drive / c[:, None]
        generator[6, 5], generator[7, 6] = 1. / duration, 2. / duration
        initial_augmented = np.concatenate((initial_hom, [1., 0., 0.]))

        def state(t):
            homogeneous = (expm(generator * (t - start)) @ initial_augmented)[:5]
            return homogeneous + np.real(np.sum(particular * np.exp(1j * omega * t)[:, None], axis=0))

        def input_power(t):
            external = np.real(np.sum(forcing * np.exp(1j * omega * t)[:, None], axis=0))
            return float(state(t) @ external)

        def heat_power(t):
            x = state(t)
            return float(x @ g @ x)

        expected_input = quad(input_power, start, start + duration, epsabs=1e-14, epsrel=1e-12)[0]
        expected_heat = quad(heat_power, start, start + duration, epsabs=1e-14, epsrel=1e-12)[0]
        energy_change = 0.5 * np.sum(c * (state(start + duration) ** 2 - state(start) ** 2))
        self.assertGreater(abs(energy_change - expected_input + expected_heat), 1e-7)

        unit = ModalBlock(np.diag(c[:2]), unit_g, "audit_polynomial_unit")
        active = ModalBlock([[c[4]]], [[local_g[4, 4]]], "audit_polynomial_active")
        members = np.array([0, 1])
        base = SimpleNamespace(model=SimpleNamespace(P=2, S=1), unit_blocks=[unit], unit_members=[members])
        audit_modal = {
            "active": (active.to_modal(particular[:, 4:]), forcing[:, 4:] @ active.phi),
            "units": {0: (unit.to_modal(particular[:, :4].reshape(2, 2, 2)),
                           forcing[:, :4].reshape(2, 2, 2) @ unit.phi)}}
        entry = {"block": active, "port_pos": np.array([0]), "port_global": np.array([2]),
                 "g_off_edges": (np.array([0, 2, 1, 2]), np.array([2, 0, 2, 1]),
                                 np.array([-0.13, -0.13, -0.19, -0.19]))}
        record = {"a": 0., "h": duration, "ya0": active.to_modal(initial_hom[4:]),
                  "a_coupling": active_coupling, "unit_coupling": unit_coupling,
                  "unit_y0": [(0, members, unit.to_modal(initial_hom[:4].reshape(2, 2)))]}
        for implementation in (MultirateSolver, ContinuousMultirateSolver):
            solver = implementation.__new__(implementation)
            solver.base, solver.row, solver._audit_row = base, 0, 0
            solver._audit_rules, solver.dwell = {}, duration
            solver.tones, solver._audit_modal = SimpleNamespace(omega=omega), audit_modal
            if implementation is ContinuousMultirateSolver:
                solver.weight_cache = BoundedWeightCache(1024**2)
                solver._audit_weights_cache = solver.weight_cache.scope("audit")
                solver.stats = dict(audit_weight_cache_hits=0, audit_weight_cache_misses=0)
            for order in (16, 32):
                with self.subTest(implementation=implementation.__name__, order=order):
                    actual = solver.upstream_energy_audit(entry, record, start, order)
                    self.assertAlmostEqual(actual["input_work"], expected_input, delta=1e-12)
                    self.assertAlmostEqual(actual["heat"], expected_heat, delta=1e-12)
                    self.assertGreater(abs(energy_change - actual["input_work"] + actual["heat"]), 1e-7)

    def test_window_excludes_next_boundary_charge_but_includes_interval_end(self):
        contributions = [(0.0, True, {"charge": 2.0}),
                         (1.0, False, {"integrated": 3.0}),
                         (1.0, True, {"charge": 5.0}),
                         (2.0, False, {"integrated": 7.0})]
        self.assertEqual(prefix_work(contributions, 1.0, ("charge", "integrated")),
                         {"charge": 2.0, "integrated": 3.0})
        self.assertEqual(prefix_work(contributions, 2.0, ("charge", "integrated")),
                         {"charge": 7.0, "integrated": 10.0})


if __name__ == "__main__":
    unittest.main()
