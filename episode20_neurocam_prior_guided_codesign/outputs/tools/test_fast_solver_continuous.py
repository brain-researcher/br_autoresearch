"""Focused checks of the new continuous adapter; execute only in Slurm."""
import gc
import math
import os
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from fast_solver_engine import JMAX
from fast_solver_multirate import MultirateSolver
from fast_solver_continuous_kernel import (BoundedWeightCache, ContinuousMultirateSolver,
                                            array_storage, coarse_bounds, dwell_spans)
from fast_solver_continuous import compute_context, window_summary


class ContinuousAdapterTests(unittest.TestCase):
    def test_cache_counts_full_backing_arrays_once_across_scopes(self):
        backing = np.empty((13, 7))
        cache = BoundedWeightCache(backing.nbytes)
        left, right = cache.scope("left"), cache.scope("right")
        left["view"] = SimpleNamespace(node_exp=backing[0])
        right["other_view"] = backing[1]
        self.assertEqual(cache.bytes, backing.nbytes)
        self.assertEqual(len(array_storage((backing[0], backing[1]))), 1)
        del left["view"]
        self.assertEqual(cache.bytes, backing.nbytes)
        del right["other_view"]
        self.assertEqual(cache.bytes, 0)

    def test_cache_eviction_is_shared_and_oversized_values_are_not_retained(self):
        cache = BoundedWeightCache(160)
        a, b = cache.scope("a"), cache.scope("b")
        a[1], b[1] = np.zeros(10), np.ones(10)
        self.assertIsNotNone(a.get(1))  # a is now most recently used.
        a[2] = np.full(10, 2.)
        self.assertIsNone(b.get(1))
        self.assertEqual(cache.bytes, 160)
        self.assertEqual(cache.evictions, 1)
        b[2] = np.zeros(100)
        self.assertIsNone(b.get(2))
        self.assertLessEqual(cache.peak_bytes, cache.max_bytes)

    def test_audit_cache_distinguishes_basis_duration_nodes_and_retains_rates(self):
        solver = ContinuousMultirateSolver.__new__(ContinuousMultirateSolver)
        solver.weight_cache = BoundedWeightCache(1024**2)
        solver._audit_weights_cache = solver.weight_cache.scope("audit")
        solver.stats = dict(audit_weight_cache_hits=0, audit_weight_cache_misses=0)
        lam = np.array([-2., -7.])
        rates_ref = weakref.ref(lam)
        h = 0.1
        nodes16 = 0.5 * (np.polynomial.legendre.leggauss(16)[0] + 1.)
        nodes32 = 0.5 * (np.polynomial.legendre.leggauss(32)[0] + 1.)
        first = solver._audit_weights(lam, h, nodes16)
        self.assertIs(first, solver._audit_weights(lam, h, nodes16.copy()))
        # Even an equal-valued distinct rate array has a separate basis key.
        self.assertIsNot(first, solver._audit_weights(lam.copy(), h, nodes16))
        adjacent_h = float(np.nextafter(h, np.inf))
        other_duration = solver._audit_weights(lam, adjacent_h, nodes16)
        self.assertIsNot(first, other_duration)
        self.assertEqual(other_duration.h.hex(), adjacent_h.hex())
        changed_nodes = nodes16.copy()
        changed_nodes[0] = np.nextafter(changed_nodes[0], np.inf)
        self.assertIsNot(first, solver._audit_weights(lam, h, changed_nodes))
        other_order = solver._audit_weights(lam, h, nodes32)
        self.assertIsNot(first, other_order)
        self.assertEqual(other_order.node_exp.shape, (2, 32))
        self.assertEqual(solver.stats, dict(audit_weight_cache_hits=1, audit_weight_cache_misses=5))
        self.assertEqual(len(solver._audit_weights_cache), 5)
        self.assertIn(id(lam), solver.weight_cache.owners)
        del lam
        self.assertIsNotNone(rates_ref())  # NodeWeights itself does not retain lam.
        # A different scope evicts audit entries under the same byte limit.
        solver.weight_cache.scope("other")["fill"] = np.zeros(solver.weight_cache.max_bytes // 8)
        self.assertEqual(len(solver._audit_weights_cache), 0)
        # Collect the temporary recursive storage-walk closure after eviction.
        gc.collect()
        self.assertIsNone(rates_ref())
        self.assertEqual(solver.weight_cache.bytes, solver.weight_cache.max_bytes)
        self.assertLessEqual(solver.weight_cache.peak_bytes, solver.weight_cache.max_bytes)

    def fine_solver(self):
        solver = ContinuousMultirateSolver.__new__(ContinuousMultirateSolver)
        nodes = np.array([0., .5, 1.])
        solver.base = SimpleNamespace(nodes=nodes, degree=2,
            vinv=np.linalg.inv(np.vander(nodes, 3, increasing=True)))
        solver.weight_cache = BoundedWeightCache(1024**2)
        solver._a_weights = solver.weight_cache.scope("fine")
        solver.stats = dict(fine_pieces=0)
        block = SimpleNamespace(lam=np.array([-2., -7.]), phi=np.eye(2))
        return solver, dict(block=block, port_pos=np.arange(2), source_pos=np.arange(2))

    def test_fine_cache_distinguishes_adjacent_float_durations(self):
        solver, entry = self.fine_solver()
        record = dict(t0=0., t1=1., h=1., a_coupling=np.zeros((2, JMAX + 1)))
        durations = (2.5e-6, np.nextafter(2.5e-6, np.inf))
        for h in durations:
            solver.base.a_y = np.array([.1, -.2])
            _, actual_h = solver.fine_piece(entry, record, 0., h)
            self.assertEqual(actual_h.hex(), float(h).hex())
            np.testing.assert_allclose(solver.base.a_y, np.array([.1, -.2]) * np.exp(entry["block"].lam * h),
                                       rtol=5e-15, atol=0.)
        self.assertEqual(len(solver._a_weights), 2)
        self.assertEqual({value.h.hex() for value in solver._a_weights.values()},
                         {float(h).hex() for h in durations})

    def test_fine_absolute_interval_preserves_duration_and_affine_forcing(self):
        solver, entry = self.fine_solver()
        t0, t1 = 3.9999375, 4.
        u0, u1 = 3.9999425, 3.999944
        record_h = t1 - t0
        forcing = np.zeros((2, JMAX + 1))
        forcing[:, :2] = [[.2, -.1], [-.3, .4]]
        record = dict(t0=t0, t1=t1, h=record_h, a_coupling=forcing)
        initial = np.array([.1, -.2])
        solver.base.a_y = initial.copy()
        _, h = solver.fine_piece(entry, record, u0, u1)
        self.assertEqual(h.hex(), (u1 - u0).hex())
        # Analytic scalar ODE y'=lambda*y+A+B*tau for the original affine drive.
        lam = entry["block"].lam
        a = forcing[:, 0] + forcing[:, 1] * ((u0 - t0) / record_h)
        b = forcing[:, 1] / record_h
        z = lam * h
        e1 = np.expm1(z) / lam
        # Series form avoids cancellation in the linear-forcing integral.
        e2 = h*h*sum(z**k / math.factorial(k + 2) for k in range(12))
        expected = np.exp(z) * initial + e1 * a + e2 * b
        np.testing.assert_allclose(solver.base.a_y, expected, rtol=5e-15, atol=1e-17)

    def test_coarse_piece_uses_actual_duration_and_restores_driver_clock(self):
        solver = ContinuousMultirateSolver.__new__(ContinuousMultirateSolver)
        block = SimpleNamespace()
        solver._prepared_blocks = {id(block)}
        solver.dwell, solver.fractions, solver.stats = 62.5e-6, (0., .5, 1.), dict(dwells=9)
        t0, t1 = 3.9999375, 3.9999375 + 1e-7

        def coarse(owner, entry, start):
            self.assertEqual(owner.dwell.hex(), (t1-t0).hex())
            self.assertEqual(owner.fractions, (0., 1.))
            self.assertEqual(start, t0)
            owner.stats["dwells"] += 1
            return [dict(a=0., b=1., h=owner.dwell, key=owner._key(owner.dwell))]

        with patch.object(MultirateSolver, "coarse_dwell", coarse):
            record = solver.coarse_piece(dict(block=block), t0, t1)
        self.assertEqual((record["t0"], record["t1"], record["h"]), (t0, t1, t1-t0))
        self.assertEqual((solver.dwell, solver.fractions, solver.stats["dwells"]), (62.5e-6, (0., .5, 1.), 9))
        with patch.object(MultirateSolver, "coarse_dwell", side_effect=ValueError("injected failure")):
            with self.assertRaisesRegex(ValueError, "injected failure"):
                solver.coarse_piece(dict(block=block), t0, t1)
        self.assertEqual((solver.dwell, solver.fractions, solver.stats["dwells"]), (62.5e-6, (0., .5, 1.), 9))

    def test_downstream_selection_ownership_survives_base_cache_replacement(self):
        solver = ContinuousMultirateSolver.__new__(ContinuousMultirateSolver)
        solver._down_selections = {}
        solver.weight_cache = BoundedWeightCache(1024)
        created = []

        def matrices(selection):
            value = dict(weights={})
            created.append(value)
            return value

        solver.base = SimpleNamespace(_down_matrices=matrices)
        first = solver.down_matrices({"adc": 0})
        for index in range(1, 70):
            solver.down_matrices({"adc": index})
        self.assertIs(first, solver.down_matrices({"adc": 0}))
        self.assertEqual(len(created), 70)
        self.assertEqual(len(solver._down_selections), 70)

    def test_frame_seam_uses_shifted_endpoints_and_restarts_only_action_index(self):
        template = SimpleNamespace(actions=50, action_row=np.repeat([0, 1], 25))
        window = SimpleNamespace(hold_action=np.arange(50), split_hold=np.repeat(np.arange(50), 3))
        expected = [(0, 25, 0, 75, 0), (25, 50, 75, 150, 1)]
        self.assertEqual(list(dwell_spans(window, template)), expected)
        first = coarse_bounds(.0039375, .004, (0., .125, .5, 1.))
        second = coarse_bounds(.004, .0040625, (0., .125, .5, 1.))
        self.assertEqual(first[-1], second[0])
        self.assertEqual(first[0], .0039375)
        self.assertEqual(second[-1], .0040625)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertTrue(np.all(np.diff(second) > 0))

    def test_energy_columns_preserve_boundary_ownership_and_telescoping(self):
        fast = SimpleNamespace(input_works=lambda start, end: {})
        contributions = [(1., True, dict(supplied_charge_boundary_work_joule=2.)),
                         (2., True, dict(supplied_charge_boundary_work_joule=99.))]
        upstream = [(2., dict(input_work=4., heat=3.), dict(input_work=4., heat=3.))]
        diagnostics = [(2., dict(sweeps=2, change=1e-12, rep_pad=0., rep_source=0.))]
        work, audit, _ = window_summary(fast, contributions, upstream, [], diagnostics, [], 1., 2., 5., 8.)
        self.assertEqual(len(work), 10)
        self.assertEqual(work["supplied_charge_boundary_work_joule"], 2.)
        self.assertEqual(work["physical_boundary_energy_change_joule"], 2.)
        self.assertEqual(work["physical_interval_energy_change_joule"], 1.)
        self.assertEqual(audit["residual"], 0.)

    def test_one_numerical_worker_records_larger_actual_cpu_allocation(self):
        from fast_io_case import THREAD_VARIABLES
        environment = {key: "1" for key in THREAD_VARIABLES}
        environment.update(SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="3", SLURM_MEM_PER_NODE="16384")
        with patch.dict(os.environ, environment), patch("fast_solver_continuous.socket.gethostname", return_value="sh01-01n01"):
            allocation = compute_context()
        self.assertEqual(allocation["allocated_cpus"], 3)
        self.assertEqual(allocation["numerical_workers"], 1)


if __name__ == "__main__":
    if "-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("compute allocation required; root owns the test session")
    unittest.main()
