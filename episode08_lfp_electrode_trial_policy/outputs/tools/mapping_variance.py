"""Acquired-feature design index; no source handles, future rows or targets."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


POLICY = "mapping_design_value"


class MappingDesign:
    """Exact rank-one algebra in fixed initializer coordinates, float64.

    C=Z0 A^-1 Z_acquired' and h=diag(Z_acquired A^-1 Z_acquired')
    avoid recomputing a large matrix solve for every direction score. The
    squared C columns give the nonnegative numerator without trace subtraction.
    """
    def __init__(self, initial_x, labels, statistics, alpha):
        labels = np.asarray(labels)
        x = np.asarray(initial_x, dtype=np.float64)
        if (x.ndim != 2 or len(x) != 32 or labels.shape != (32,)
                or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, 4))
                or not np.isfinite(x).all() or not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("Exactly four acquired initializer rows per direction required")
        self.direction_mean = np.array(statistics.direction_input_mean, copy=True)
        self.mean = np.array(statistics.input_mean, copy=True)
        self.scale = np.array(statistics.input_sd, copy=True)
        if (self.direction_mean.shape != (8, x.shape[1]) or self.mean.shape != (x.shape[1],)
                or self.scale.shape != self.mean.shape or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError("Fixed finite initializer coordinates required")
        self.z0 = self.coordinates(x, labels)
        self.z = self.z0.copy()
        self.labels = labels.copy()
        self.eta = float(32 * alpha)
        a = self.z0.T @ self.z0 + self.eta * np.eye(x.shape[1])
        self.inverse = cho_solve(cho_factor(a, lower=True), np.eye(x.shape[1]))
        self.inverse = (self.inverse + self.inverse.T) / 2
        projection = self.z0 @ self.inverse
        self.cross = projection @ self.z0.T
        self.leverage = np.einsum("ij,ij->i", projection, self.z0)

    def coordinates(self, values, labels):
        values, labels = np.asarray(values, dtype=np.float64), np.asarray(labels)
        if (values.ndim != 2 or values.shape[1] != len(self.scale)
                or labels.shape != (len(values),) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels > 7) or not np.isfinite(values).all()):
            raise ValueError("Finite already acquired rows and direction indices required")
        return (values - self.direction_mean[labels] - self.mean) / self.scale

    def choose(self, counts):
        counts = np.asarray(counts)
        if (counts.shape != (8,) or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 4) or np.any(counts > 12)
                or not np.array_equal(counts, np.bincount(self.labels, minlength=8))):
            raise ValueError("Counts must describe only this acquired history")
        legal = np.flatnonzero(counts < 12)
        if not len(legal):
            raise ValueError("Finite queue exhausted")
        denominator = 1 + self.leverage
        gains = np.sum(self.cross * self.cross, axis=0) / (32 * denominator)
        if np.any(denominator <= 0) or not np.isfinite(gains).all():
            raise ArithmeticError("Invalid acquired design index; no jitter or clipping")
        utilities = np.array([gains[self.labels == d].mean() for d in range(8)])
        chosen = int(min(legal, key=lambda d: (-utilities[d], counts[d], d)))
        return chosen, {"counts_before": counts.tolist(), "utility": utilities.tolist(),
                        "design_index_before": float(np.trace(self.cross[:, :32]) / 32)}

    def observe(self, row, direction):
        """Called only after the trusted trajectory reveals the chosen row."""
        if type(direction) is not int or direction not in range(8):
            raise ValueError("A chosen direction is required")
        z = self.coordinates(np.asarray(row)[None, :], np.array([direction]))[0]
        v = self.inverse @ z
        h, c, q = self.z @ v, self.z0 @ v, float(z @ v)
        denominator = 1 + q
        if denominator <= 0 or not np.isfinite(denominator):
            raise ArithmeticError("Invalid rank-one update; no numerical retuning")
        self.cross = np.column_stack((self.cross - np.outer(c, h) / denominator, c / denominator))
        self.leverage = np.append(self.leverage - h * h / denominator, q / denominator)
        self.inverse -= np.outer(v, v) / denominator
        self.z = np.vstack((self.z, z))
        self.labels = np.append(self.labels, direction)
        if not np.isfinite(self.cross).all() or not np.isfinite(self.inverse).all():
            raise ArithmeticError("Nonfinite design update")
