"""One prospective continuous representation for the fixed selected locations.

All classes use identical native values and this anatomy-only interpolation
rule. Off-grid values are model extrapolations, not measured biological truth.
No image reading, support expansion, smoothing, bandwidth search or ridge.
"""
import numpy as np

from local_functional_source import _array, _frozen
from development_lopo import _support_size


class CommonCardinalSource:
    """Constant-augmented Gaussian cardinal interpolation, selected pre-score.

    Bandwidth is the median nearest-neighbor distance on the exact common
    support. A single augmented system defines weights and derivatives for
    every class/variant. Native queries have literal Kronecker weights, while
    derivatives always come from the analytic cardinal function, even there.
    """

    def __init__(self, coordinates, voxel_ids, *, numerical_atol=1e-10,
                 rank_rtol=1e-8, amplification_cap=4., support_size=208):
        self.support_size = _support_size(support_size)
        if (numerical_atol != 1e-10 or rank_rtol != 1e-8 or amplification_cap != 4.
                or isinstance(amplification_cap, (bool, np.bool_))):
            raise ValueError("the selected cardinal numerical recipe must remain fixed")
        self.coordinates = _frozen(_array(coordinates, (self.support_size, 3), "common coordinates in mm"))
        if (not isinstance(voxel_ids, tuple) or len(voxel_ids) != self.support_size
                or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in voxel_ids)
                or len(set(voxel_ids)) != self.support_size):
            raise ValueError("exact ordered unique native integer voxel identities are required")
        self.voxel_ids = voxel_ids
        self.atol, self.rank_rtol, self.amplification_cap = numerical_atol, rank_rtol, amplification_cap
        delta = self.coordinates[:, None]-self.coordinates[None]
        squared = np.sum(delta*delta, axis=-1)
        distances = np.sqrt(squared)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        if np.any(nearest <= 0):
            raise ValueError("distinct support coordinates are required; none may be removed")
        self.bandwidth_mm = float(np.median(nearest))
        kernel = np.exp(-squared/(2*self.bandwidth_mm**2))
        augmented = np.block([[kernel, np.ones((self.support_size, 1))],
                              [np.ones((1, self.support_size)), np.zeros((1, 1))]])
        singular = np.linalg.svd(augmented, compute_uv=False)
        if not np.isfinite(singular).all() or singular[-1] <= rank_rtol*singular[0]:
            raise ValueError("common cardinal system fails the fixed relative singular cutoff")
        self.relative_smallest_singular = float(singular[-1]/singular[0])
        rhs = np.vstack((np.eye(self.support_size), np.zeros((1, self.support_size))))
        coefficients = np.linalg.solve(augmented, rhs)
        if (not np.isfinite(coefficients).all()
                or np.max(np.abs(augmented@coefficients-rhs)) > self.atol):
            raise ArithmeticError("cardinal solve failed identity/constant-system accuracy")
        self.coefficients = _frozen(coefficients)
        # Check the analytic interpolant before enabling exact-node evaluation.
        raw, derivative = self._analytic(self.coordinates)
        if (np.max(np.abs(raw-np.eye(self.support_size))) > self.atol
                or np.max(np.abs(raw.sum(axis=1)-1)) > self.atol
                or np.max(np.abs(derivative.sum(axis=1))) > self.atol):
            raise ArithmeticError("analytic cardinal identity or constant reproduction failed")

    def _queries(self, queries):
        raw = np.asarray(queries)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
            raise ValueError("queries must be nonempty common-space XYZ rows")
        return _array(raw, raw.shape, "query coordinates in mm")

    def _analytic(self, queries):
        delta = queries[:, None]-self.coordinates[None]
        kernel = np.exp(-np.sum(delta*delta, axis=-1)/(2*self.bandwidth_mm**2))
        weights = kernel@self.coefficients[:-1]+self.coefficients[-1]
        kernel_gradient = -kernel[..., None]*delta/self.bandwidth_mm**2
        derivative = np.einsum("qkd,ki->qid", kernel_gradient, self.coefficients[:-1])
        return weights, derivative

    def weights_and_derivatives(self, queries):
        queries = self._queries(queries)
        weights, derivative = self._analytic(queries)
        matches = np.all(queries[:, None] == self.coordinates[None], axis=-1)
        rows, columns = np.nonzero(matches)
        weights[rows] = 0.
        weights[rows, columns] = 1.
        if not np.isfinite(weights).all() or not np.isfinite(derivative).all():
            raise ArithmeticError("nonfinite cardinal evaluation")
        # The shortcut changes value evaluation only. It must not zero the
        # analytic derivative used by an optimizer starting at identity.
        return weights, derivative

    def certificate(self, weights):
        raw = np.asarray(weights)
        if raw.ndim != 2 or raw.shape[1] != self.support_size or len(raw) == 0:
            raise ValueError("weights must retain the complete source support")
        W = _array(raw, raw.shape, "cardinal weights")
        constant_error = float(np.max(np.abs(W.sum(axis=1)-1)))
        amplification = float(np.max(np.sum(np.abs(W), axis=1)))
        # Centering prediction rows corresponds to centering W over queries.
        centered_amplification = float(np.max(np.sum(np.abs(W-W.mean(axis=0)), axis=1)))
        if constant_error > self.atol or amplification > self.amplification_cap+self.atol:
            raise ValueError("cardinal interpolation exceeds fixed constant/amplification bounds")
        return {"constant_reproduction_max_abs": constant_error,
                "interpolation_amplification": amplification,
                "centered_interpolation_amplification": centered_amplification,
                "interpolation_amplification_cap": self.amplification_cap,
                "interpolation_bandwidth_mm": self.bandwidth_mm,
                "interpolation_relative_smallest_singular": self.relative_smallest_singular,
                "off_grid_biological_accuracy_certified": False}

    def evaluate(self, maps, queries):
        """Evaluate supplied map rows without centering, gain or response fitting."""
        raw = np.asarray(maps)
        if raw.ndim != 2 or raw.shape[1] != self.support_size or len(raw) == 0:
            raise ValueError("map rows must use the complete source support")
        values = _array(raw, raw.shape, "source maps")
        W, derivative = self.weights_and_derivatives(queries)
        self.certificate(W)
        prediction = values@W.T
        gradients = np.einsum("ci,qid->cqd", values, derivative)
        if not np.isfinite(prediction).all() or not np.isfinite(gradients).all():
            raise ArithmeticError("nonfinite source-map interpolation")
        return prediction, gradients
