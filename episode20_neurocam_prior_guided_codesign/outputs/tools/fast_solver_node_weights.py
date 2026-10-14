"""Endpoint/node-only counterpart of fast_solver_phi.PieceWeights.

The continuous solver uses this helper. It uses the existing
phi_table implementation and the same operation order for endpoint and node
response weights, while omitting unused moment arrays and their computation.
There is no cache, duration normalization, node change or new approximation.
"""
import numpy as np

from fast_solver_phi import _FACT, phi_table


class NodeWeights:
    """Response weights for y' = lam*y + sum_j forcing_j*(tau/h)**j.

    Provides end_exp, end_force, node_exp and node_force with the same shapes
    and arithmetic as PieceWeights. Callers needing moments must continue to
    use PieceWeights. Rates, h, jmax and nodes retain the original meanings.
    """

    def __init__(self, lam, h, jmax, nodes):
        lam = np.asarray(lam, dtype=float)
        if np.any(lam > 0) or not np.all(np.isfinite(lam)):
            raise ValueError("modal rates must be finite and nonpositive")
        self.h, self.jmax = float(h), int(jmax)
        self.nodes = np.asarray(nodes, dtype=float)
        z = lam * self.h
        self.z = z
        phis = phi_table(z, jmax + 1)
        self.end_exp = phis[0]
        j = np.arange(jmax + 1)
        self.end_force = self.h * phis[1:jmax + 2].T * _FACT[j][None, :]
        zs = z[:, None] * self.nodes[None, :]
        node_phi = phi_table(zs, jmax + 1)
        self.node_exp = node_phi[0]
        powers = self.nodes[None, :] ** (j[:, None] + 1)
        self.node_force = self.h * np.transpose(node_phi[1:jmax + 2], (1, 0, 2)) * (
            _FACT[j][None, :, None] * powers[None, :, :])
