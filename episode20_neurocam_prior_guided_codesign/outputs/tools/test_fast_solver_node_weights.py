"""Focused equivalence tests for the inactive endpoint/node-only helper.

Run in root's compute allocation. These kernel tests read no episode payloads
and neither import nor run either scientific driver.
"""
import os
import socket
import unittest

import numpy as np

from fast_solver_node_weights import NodeWeights
from fast_solver_phi import PieceWeights, chebyshev_lobatto, shift_monomials


def responses(weights, initial, forcing):
    endpoint = weights.end_exp*initial + np.einsum("kj,kj->k", forcing, weights.end_force)
    nodes = weights.node_exp*initial[:, None] + np.einsum("kj,kji->ki", forcing, weights.node_force)
    return endpoint, nodes


class NodeWeightsTests(unittest.TestCase):
    def assert_weights_equal(self, rates, duration, degree, nodes):
        expected = PieceWeights(rates, duration, degree, nodes, 10)
        actual = NodeWeights(rates, duration, degree, nodes)
        for name in ("z", "nodes", "end_exp", "end_force", "node_exp", "node_force"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)
        self.assertFalse(hasattr(actual, "mom_exp"))
        self.assertFalse(hasattr(actual, "mom_force"))
        return expected, actual

    def test_zero_small_transition_and_stiff_modes_on_original_node_grids(self):
        grids = [chebyshev_lobatto(8)]
        for order in (16, 32):
            nodes, _ = np.polynomial.legendre.leggauss(order)
            grids.append(.5*(nodes+1.))
        # Include both sides of the existing scaling and moment switches,
        # plus a strongly stiff mode. No helper-specific numerical thresholds.
        dimensionless_rates = np.array([0., -1e-16, -1e-8, -.499999, -.500001,
                                        -39.999, -40.001, -1e4, -1e10])
        for duration in (2.5e-6, 62.5e-6, 62.5e-6/4096):
            for nodes in grids:
                with self.subTest(duration=duration, node_count=len(nodes)):
                    self.assert_weights_equal(dimensionless_rates/duration, duration, 10, nodes)

    def test_arbitrary_polynomial_forcing_responses(self):
        rates = np.array([0., -1e-12, -1., -1e5, -1e9, -1e15])
        initial = np.array([.2, -.4, 0., 1.3, -.8, .01])
        for degree in (0, 8, 10):
            # Dense signed coefficients exercise every forcing column.
            index = np.arange(len(rates)*(degree+1)).reshape(len(rates), degree+1)
            forcing = ((index % 11)-5.)/7.
            expected, actual = self.assert_weights_equal(rates, 2.5e-6, degree, chebyshev_lobatto(8))
            for new, old in zip(responses(actual, initial, forcing), responses(expected, initial, forcing)):
                np.testing.assert_array_equal(new, old)

    def test_partial_intervals_with_reexpressed_polynomial_forcing(self):
        rates = np.array([0., -1e-8, -1e3, -1e8, -1e14])
        duration = 62.5e-6
        degree = 10
        initial = np.array([-.3, .4, -.7, 0., .02])
        index = np.arange(len(rates)*(degree+1)).reshape(len(rates), degree+1)
        forcing = ((index % 13)-6.)/9.
        nodes = np.array([0., .07, .5, .93, 1.])
        for start, stop in ((0., 1./4096), (.125, .3125), (.7, .999999)):
            with self.subTest(start=start, stop=stop):
                width = stop-start
                restricted = shift_monomials(forcing, start, width)
                expected, actual = self.assert_weights_equal(rates, duration*width, degree, nodes)
                for new, old in zip(responses(actual, initial, restricted), responses(expected, initial, restricted)):
                    np.testing.assert_array_equal(new, old)

    def test_zero_mode_matches_polynomial_antiderivative(self):
        duration = 2.5e-6
        nodes = np.array([0., .1, .5, .9, 1.])
        forcing = np.array([[.75, -.3, 1.2, -.45]])
        weights = NodeWeights(np.array([0.]), duration, 3, nodes)
        endpoint, values = responses(weights, np.array([0.]), forcing)
        powers = np.arange(1, forcing.shape[1]+1)
        exact = duration*np.sum(forcing[0, :, None]*nodes[None, :]**powers[:, None]/powers[:, None], axis=0)
        np.testing.assert_allclose(values[0], exact, rtol=5e-15, atol=0.)
        np.testing.assert_allclose(endpoint[0], exact[-1], rtol=5e-15, atol=0.)


if __name__ == "__main__":
    if "-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("compute allocation required; root owns the test session")
    unittest.main()
