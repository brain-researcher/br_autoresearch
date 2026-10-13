#!/usr/bin/env python3
"""Tiny candidate-array reproducer for the rounded-moment derivative defect.

These artificial three/four-candidate arrays exercise arithmetic only.  They
are not episode qualification data and run no optimizer or fitted experiment.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import fit_v3_m1 as m1


class CandidateArrayGeometry:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw
        self.calls: list[tuple[int, int]] = []

    def raw_batch(self, cell, start: int, stop: int) -> np.ndarray:
        self.calls.append((start, stop))
        return self.raw[start:stop]


def fixture(contrasts: np.ndarray, raw: np.ndarray | None = None):
    count = len(contrasts)
    cell = SimpleNamespace(k=1, observed_index=count - 1)
    x = np.zeros((1, m1.DESIGN_DIMENSION))
    x[0, 0] = 1.0
    prepared = m1.PreparedData(
        (cell,), x, np.zeros((1, m1.GEOMETRY_DIMENSION)),
        np.ones(m1.GEOMETRY_DIMENSION), np.ones(1), None,
    )
    basis = m1.CandidateBasis(1, np.zeros((count, m1.PANEL_SIZE)), contrasts)
    library = SimpleNamespace(basis=lambda k: basis)
    geometry = CandidateArrayGeometry(
        np.zeros((count, m1.GEOMETRY_DIMENSION)) if raw is None else raw
    )
    return prepared, library, geometry


class DirectDerivativeRegressionTests(unittest.TestCase):
    def test_center_before_expectation_preserves_rounded_away_signal(self) -> None:
        contrasts = np.zeros((3, m1.TARGET_CONTRAST_DIMENSION))
        contrasts[:, 0] = [math.nextafter(1.0, math.inf), 1.0, 1.0]
        prepared, library, geometry = fixture(contrasts)
        theta = np.zeros(m1.PARAMETER_DIMENSION)
        direction = theta.copy()
        direction[0] = 1.0
        expected = (math.nextafter(1.0, math.inf) - 1.0) / 3.0
        with mock.patch.object(m1, "BATCH_MAXIMUM", 2):
            ordinary = m1.evaluate_model(
                theta, prepared, library, geometry, 0.0, need_hessian=False
            )
            direct, visits = m1._direct_directional_derivative(
                prepared, library, geometry, theta, direction, 0.0
            )
            line = m1.evaluate_model(
                theta, prepared, library, geometry, 0.0, need_hessian=False,
                line_origin=theta, line_direction=direction, line_step=0.0,
            )
        # The original mean-minus-observed pathway really loses this signal.
        self.assertEqual(ordinary.gradient[0], 0.0)
        self.assertGreater(expected, 0.0)
        self.assertEqual(direct, expected)
        self.assertEqual(line.direct_directional_derivative, expected)
        self.assertEqual(visits, 6)
        self.assertEqual(line.candidate_visits, 6)
        self.assertTrue(all(stop - start <= 2 for start, stop in geometry.calls))
        self.assertEqual(line.objective, ordinary.objective)
        np.testing.assert_array_equal(line.gradient, ordinary.gradient)

    def test_initial_and_trial_match_independent_full_feature_expectation(self) -> None:
        contrasts = np.zeros((4, m1.TARGET_CONTRAST_DIMENSION))
        contrasts[:, :2] = [[0.3, -0.1], [0.2, 0.5], [-0.4, 0.1], [0.1, -0.2]]
        raw = np.zeros((4, m1.GEOMETRY_DIMENSION))
        raw[:, :2] = [[0.2, -0.1], [0.5, 0.4], [-0.1, -0.2], [0.3, 0.1]]
        prepared, library, geometry = fixture(contrasts, raw)
        prepared.x[0, 1] = -0.25
        theta = 0.01 * np.cos(np.arange(m1.PARAMETER_DIMENSION))
        direction = 0.03 * np.sin(np.arange(m1.PARAMETER_DIMENSION) + 0.4)
        correction = 0.002 * np.cos(np.arange(m1.PARAMETER_DIMENSION) + 0.7)
        ridge = 0.2
        full_features = np.asarray([
            np.concatenate((np.outer(prepared.x[0], a).ravel(), z))
            for a, z in zip(contrasts, raw, strict=True)
        ])
        for step in (0.0, 0.25):
            candidate = theta + step * direction
            scores = [math.fsum((candidate * feature).tolist()) for feature in full_features]
            weights = [math.exp(score - max(scores)) for score in scores]
            increments = [
                math.fsum((direction * (feature - full_features[-1])).tolist())
                for feature in full_features
            ]
            expected = math.fsum([
                math.fsum(w * d for w, d in zip(weights, increments, strict=True))
                / math.fsum(weights),
                ridge * math.fsum((candidate * direction).tolist()),
                math.fsum((correction * direction).tolist()),
            ])
            with mock.patch.object(m1, "BATCH_MAXIMUM", 2):
                direct, visits = m1._direct_directional_derivative(
                    prepared, library, geometry, candidate, direction, ridge, correction
                )
                line = m1.evaluate_model(
                    candidate, prepared, library, geometry, ridge, need_hessian=False,
                    linear_correction=correction, line_origin=theta,
                    line_direction=direction, line_step=step,
                )
                plain = m1.evaluate_model(
                    candidate, prepared, library, geometry, ridge,
                    need_hessian=False, linear_correction=correction,
                )
            self.assertAlmostEqual(direct, expected, delta=1.0e-16)
            self.assertEqual(line.direct_directional_derivative, direct)
            self.assertEqual(visits, 8)
            self.assertEqual(line.objective, plain.objective)
            np.testing.assert_array_equal(line.gradient, plain.gradient)

    def test_singletons_have_exact_zero_data_terms(self) -> None:
        prepared, library, geometry = fixture(np.zeros((1, m1.TARGET_CONTRAST_DIMENSION)))
        prepared.cells[0].k = 0
        prepared.cells[0].observed_index = 0
        theta = np.zeros(m1.PARAMETER_DIMENSION)
        direction = theta.copy()
        correction = theta.copy()
        theta[0], direction[0], correction[0] = 0.5, 0.25, -0.125
        for k in (0, m1.PANEL_SIZE):
            prepared.cells[0].k = k
            direct, visits = m1._direct_directional_derivative(
                prepared, library, geometry, theta, direction, 2.0, correction
            )
            line = m1.evaluate_model(
                theta, prepared, library, geometry, 2.0, need_hessian=False,
                linear_correction=correction, line_origin=theta,
                line_direction=direction, line_step=0.0,
            )
            self.assertEqual(direct, 0.21875)
            self.assertEqual(line.direct_directional_derivative, direct)
            self.assertEqual(visits, 2)
            self.assertEqual(geometry.calls, [])

    def test_constant_candidate_increment_has_exact_zero_data_term(self) -> None:
        contrasts = np.ones((3, m1.TARGET_CONTRAST_DIMENSION))
        prepared, library, geometry = fixture(contrasts)
        theta = np.ones(m1.PARAMETER_DIMENSION) * 0.01
        direction = np.ones(m1.PARAMETER_DIMENSION) * 0.03
        direct, _ = m1._direct_directional_derivative(
            prepared, library, geometry, theta, direction, 0.0
        )
        self.assertEqual(direct, 0.0)

    def test_cell_ridge_sum_retains_cancellation_remainder(self) -> None:
        theta = np.zeros(m1.PARAMETER_DIMENSION)
        direction = theta.copy()
        theta[0], direction[0] = -1.0e16, 1.0
        self.assertEqual(
            m1._finish_directional_derivative([1.0e16, 1.0], theta, direction, 1.0),
            1.0,
        )

    def test_nonfinite_candidate_contribution_fails_closed(self) -> None:
        with self.assertRaises(m1.M1Error):
            m1._directional_batch_numerator(
                np.zeros((1, 21)), np.zeros(21), np.ones(21), np.asarray([math.nan])
            )


if __name__ == "__main__":
    unittest.main()
