"""Focused contact inverse checks; execution belongs in coordinated Slurm."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, diags

import radau_contact_preconditioner as module
from radau_contact_preconditioner import prepare_contact_blocks
from sparse_array_rc import PassiveStarConductance, SolverConvergenceError, _ConductanceWithStars


def coupled_fixture():
    # Three independent local contact groups, with different global neighbours.
    # The first two principal blocks are equal; the third has different loading.
    # External C/G edges must retain their diagonal contribution inside each
    # principal block even though their off-block entries are excluded from M.
    size = 54
    capacitance, base = np.eye(size) * .4, np.eye(size) * .25
    groups = []

    def edge(matrix, first, second, value):
        direction = np.zeros(size)
        direction[first], direction[second] = 1., -1.
        matrix += value * np.outer(direction, direction)

    for contact in range(3):
        nodes = tuple(range(17 * contact, 17 * (contact + 1)))
        for branch in range(1, 17):
            edge(capacitance, nodes[0], nodes[branch], .1 + .01 * branch)
        edge(capacitance, nodes[0], 51 + contact, .03 if contact < 2 else .07)
        edge(base, nodes[0], 51 + contact, .05 if contact < 2 else .09)
        groups.append((nodes, tuple(.2 + .05 * index for index in range(17)), .8))
    stars = PassiveStarConductance(state_count=size, groups=groups)
    operator = _ConductanceWithStars(csr_matrix(base), stars)
    # Apply the original centered operator to basis vectors, independently of
    # the helper's private coefficient extraction and block construction.
    dense_g = base + stars @ np.eye(size)
    return csr_matrix(capacitance), operator, capacitance, dense_g


def nodewise_reference(full_matrix, size):
    def apply(value):
        result = np.empty(2 * size)
        for node in range(size):
            indices = np.array([node, node + size])
            result[indices] = np.linalg.solve(full_matrix[np.ix_(indices, indices)], value[indices])
        return result
    return apply


class RadauContactPreconditionerTests(unittest.TestCase):
    def test_grouped_inverse_matches_dense_principal_blocks_and_exact_step_cache(self):
        c, conductance, dense_c, dense_g = coupled_fixture()
        info = {}
        plan = prepare_contact_blocks(c, conductance, max_workspace_bytes=8 * 1024**2,
                                      check=lambda: None, fallback_info=info)
        self.assertTrue(info["supported"])
        self.assertEqual(plan.metadata["contact_count"], 3)
        self.assertEqual(plan.metadata["distinct_block_count"], 2)
        size, step = c.shape[0], .23
        tableau = np.array([[5. / 12., -1. / 12.], [.75, .25]])
        full = np.kron(np.eye(2), dense_c) + step * np.kron(tableau, dense_g)
        rhs = np.sin(np.arange(2 * size) + .3)
        original_rhs = rhs.copy()
        expected = np.empty_like(rhs)
        for nodes in (*[np.arange(17 * index, 17 * (index + 1)) for index in range(3)],
                      *[np.array([index]) for index in range(51, size)]):
            indices = np.r_[nodes, nodes + size]
            expected[indices] = np.linalg.solve(full[np.ix_(indices, indices)], rhs[indices])
        fallback = nodewise_reference(full, size)
        inverse = plan.for_step(step, fallback=fallback, check=lambda: None)
        with patch.object(module, "lu_solve", wraps=module.lu_solve) as solve:
            actual = inverse @ rhs
        self.assertEqual(solve.call_count, 2)  # Equal contacts share a RHS batch.
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        np.testing.assert_array_equal(rhs, original_rhs)
        self.assertEqual(plan.factorizations, 2)
        self.assertLess(plan.numeric_storage_bytes, plan.estimated_workspace_bytes)
        repeated = plan.for_step(step, fallback=fallback, check=lambda: None)
        np.testing.assert_allclose(repeated @ rhs, expected, rtol=3e-12, atol=3e-12)
        self.assertEqual(plan.factorizations, 2)
        # An adjacent representable h is distinct; no rounding/quantization.
        plan.for_step(np.nextafter(step, np.inf), fallback=fallback, check=lambda: None)
        self.assertEqual(plan.factorizations, 4)
        with self.assertRaisesRegex(ValueError, "stale contact"):
            inverse @ rhs

    def test_unsupported_topologies_fallback_and_resource_shortfall_refuses(self):
        c, conductance, _, _ = coupled_fixture()
        with patch.object(module, "lu_factor", side_effect=AssertionError("no factor before admission")):
            with self.assertRaisesRegex(SolverConvergenceError, "numeric-workspace"):
                prepare_contact_blocks(c, conductance, max_workspace_bytes=1, check=lambda: None)
        cases = (
            (34, ((tuple(range(17)), (1.,) * 17, .1),
                  (tuple(range(16, 33)), (1.,) * 17, .1)), "overlapping"),
            # Total term count is still2*17: group membership itself must be
            # checked, not merely the average number of nodes per group.
            (34, ((tuple(range(16)), (1.,) * 16, .1),
                  (tuple(range(16, 34)), (1.,) * 18, .1)), "seventeen_node"))
        for size, groups, reason in cases:
            stars = PassiveStarConductance(state_count=size, groups=groups)
            base = diags(np.ones(size), format="csr")
            info = {}
            with self.subTest(reason=reason):
                self.assertIsNone(prepare_contact_blocks(base, _ConductanceWithStars(base, stars),
                    max_workspace_bytes=8 * 1024**2, check=lambda: None, fallback_info=info))
                self.assertFalse(info["supported"])
                self.assertIn(reason, info["reason"])
        size = 33 * 17
        c = diags(np.repeat(np.arange(33) + 1., 17), format="csr")
        base = diags(np.ones(size), format="csr")
        stars = PassiveStarConductance(state_count=size,
            groups=tuple((tuple(range(17 * index, 17 * (index + 1))), (1.,) * 17, .1)
                         for index in range(33)))
        info = {}
        self.assertIsNone(prepare_contact_blocks(c, _ConductanceWithStars(base, stars),
            max_workspace_bytes=8 * 1024**2, check=lambda: None, fallback_info=info))
        self.assertEqual(info["reason"], "more_than_32_distinct_contact_blocks")

    def test_callbacks_nonfinite_inputs_and_factor_failure_cannot_return_an_inverse_result(self):
        c, conductance, _, _ = coupled_fixture()
        plan = prepare_contact_blocks(c, conductance, max_workspace_bytes=8 * 1024**2, check=lambda: None)
        stopped = RuntimeError("owned deadline")

        def stop():
            raise stopped

        with self.assertRaises(RuntimeError) as error:
            plan.for_step(.1, fallback=lambda value: value, check=stop)
        self.assertIs(error.exception, stopped)
        calls = []

        def interrupt_application():
            if calls:
                raise stopped

        inverse = plan.for_step(.1, fallback=lambda value: value, check=interrupt_application)
        calls.append(True)
        rhs = np.ones(2 * c.shape[0])
        before = rhs.copy()
        with self.assertRaises(RuntimeError) as error:
            inverse @ rhs
        self.assertIs(error.exception, stopped)
        np.testing.assert_array_equal(rhs, before)
        inverse = plan.for_step(.1, fallback=lambda value: value, check=lambda: None)
        bad = rhs.copy()
        bad[1] = np.nan
        with self.assertRaises(ValueError):
            inverse @ bad
        empty = csr_matrix((17, 17))
        zero_star = PassiveStarConductance(state_count=17, groups=((tuple(range(17)), (0.,) * 17, 0.),))
        singular = prepare_contact_blocks(empty, _ConductanceWithStars(empty, zero_star),
            max_workspace_bytes=8 * 1024**2, check=lambda: None)
        with self.assertRaisesRegex(SolverConvergenceError, "singular contact"):
            singular.for_step(.1, fallback=lambda value: value, check=lambda: None)
        self.assertIsNone(singular.metadata["cached_step_seconds"])


if __name__ == "__main__":
    unittest.main()
