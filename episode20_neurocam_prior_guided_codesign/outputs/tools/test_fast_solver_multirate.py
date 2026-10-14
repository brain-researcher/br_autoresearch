"""Focused development checks for the unqualified dwell path; run in Slurm."""
from types import SimpleNamespace
import unittest

import numpy as np

from fast_solver_engine import JMAX
from fast_solver_multirate import MultirateSolver
from test_fast_solver_core import oracle_interval


class ActiveBlockCacheTests(unittest.TestCase):
    def test_distinct_blocks_use_their_own_modes_and_reuse_weights(self):
        nodes = np.array([0.0, 0.5, 1.0])
        base = SimpleNamespace(nodes=nodes, degree=2,
                               vinv=np.linalg.inv(np.vander(nodes, 3, increasing=True)))
        solver = MultirateSolver.__new__(MultirateSolver)
        solver.base = base
        solver._a_weights = {}
        solver.stats = {"fine_pieces": 0}
        blocks = [SimpleNamespace(lam=np.array(values), phi=np.eye(2))
                  for values in ((-2.0, -7.0), (-3.0, -11.0))]
        initial = np.array([0.002, -0.003])
        forcing = np.zeros((2, JMAX + 1))
        forcing[:, :2] = [[0.001, -0.002], [0.003, 0.004]]
        record = {"h": 0.125, "key": ("same_time_grid",), "a_coupling": forcing}
        weights_after_first_visit = {}
        for block_index in (0, 1, 0, 1):
            block = blocks[block_index]
            entry = {"block": block, "port_pos": np.arange(2), "source_pos": np.arange(2)}
            base.a_y = initial.copy()
            _, duration = solver.a_fine(entry, record, 0.0, 1.0)
            expected, _ = oracle_interval(np.eye(2), np.zeros((2, 2)), np.diag(-block.lam),
                                         initial, forcing[:, :2], record["h"])
            np.testing.assert_allclose(base.a_y, expected, rtol=0.0, atol=1e-13)
            self.assertEqual(duration, record["h"])
            cache_objects = {id(value) for value in solver._a_weights.values()}
            if block_index not in weights_after_first_visit:
                previous = set().union(*weights_after_first_visit.values()) if weights_after_first_visit else set()
                weights_after_first_visit[block_index] = cache_objects - previous
                self.assertEqual(len(weights_after_first_visit[block_index]), 1)
            else:
                self.assertEqual(cache_objects, set().union(*weights_after_first_visit.values()))
        self.assertEqual(len(solver._a_weights), 2)


if __name__ == "__main__":
    unittest.main()
