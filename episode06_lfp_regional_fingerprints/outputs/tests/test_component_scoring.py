"""Focused registered-component arithmetic checks; execute only on compute."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import component_scoring as cs


def design(**changes):
    fields = dict(session_id="fixture_session", role="evaluation", directions=("right", "left"),
                  trial_ids=("r1", "r2", "l1"), trial_directions=("right", "right", "left"),
                  time_ids=(5, 6), axis_ids=("frozen_axis_1", "frozen_axis_2"))
    fields.update(changes)
    return cs.FrozenDesign(**fields)


def bound(d, value):
    return cs.BoundActivity(d, value)


def scored(d, *, region="M1", component="C", observed=None, correct=None,
           pooled=None, swapped=None, energy_gate=0.01):
    observed = np.ones(d.shape) if observed is None else observed
    correct = observed if correct is None else correct
    pooled = np.zeros(d.shape) if pooled is None else pooled
    swapped = np.zeros(d.shape) if swapped is None else swapped
    return cs.score_component(d, region=region, component=component, energy_gate=energy_gate,
                              observed=bound(d, observed), correct=bound(d, correct),
                              pooled=bound(d, pooled), swapped=bound(d, swapped))


class ComponentScoringTests(unittest.TestCase):
    def test_independent_weighted_matrix_oracle_orthogonality_and_energy(self):
        d = design()
        y = np.arange(12, dtype=float).reshape(d.shape) - 3.0
        # Independently form the C/F indicator matrices and diagonal W.
        c = np.tile(np.eye(2), (3, 1))
        f = np.zeros((6, 4))
        f[0:2, 0:2] = np.eye(2)
        f[2:4, 0:2] = np.eye(2)
        f[4:6, 2:4] = np.eye(2)
        w = np.diag([1/8, 1/8, 1/8, 1/8, 1/4, 1/4])
        pc = c @ np.linalg.solve(c.T @ w @ c, c.T @ w)
        pf = f @ np.linalg.solve(f.T @ w @ f, f.T @ w)
        matrices = (pc, pf - pc, np.eye(6) - pf)
        pieces = []
        for name, matrix in zip("CDE", matrices):
            piece = cs.project_component(d, bound(d, y), component=name)
            np.testing.assert_allclose(piece.reshape(6, 2), matrix @ y.reshape(6, 2), atol=1e-13)
            np.testing.assert_allclose(matrix @ matrix, matrix, atol=1e-13)
            np.testing.assert_allclose(matrix.T @ w, w @ matrix, atol=1e-13)
            pieces.append(piece.reshape(6, 2))
        for i in range(3):
            for j in range(i):
                self.assertAlmostEqual(float(np.sum(pieces[i] * (w @ pieces[j]))), 0.0)
        np.testing.assert_allclose(sum(pieces), y.reshape(6, 2))
        total = float(np.sum(y.reshape(6, 2) * (w @ y.reshape(6, 2))))
        self.assertAlmostEqual(sum(float(np.sum(p * (w @ p))) for p in pieces), total)

    def test_imbalance_invariance_constant_level_and_axis_sum(self):
        d = design()
        y = np.array([[[2., 3.], [4., 5.]], [[2., 3.], [4., 5.]],
                      [[8., 9.], [10., 11.]]])
        expanded = design(trial_ids=("r1", "r2", "r3", "r4", "l1"),
                          trial_directions=("right", "right", "right", "right", "left"))
        yy = y[[0, 1, 0, 1, 2]]
        for name in "CD":
            original = scored(d, component=name, observed=y)
            repeated = scored(expanded, component=name, observed=yy)
            self.assertEqual(original.normalized_target_energy, repeated.normalized_target_energy)
            self.assertEqual(original.rules, repeated.rules)
        common = cs.project_component(d, bound(d, y), component="C")
        np.testing.assert_array_equal(common[0], [[5., 6.], [7., 8.]])
        constant = scored(d, observed=np.full(d.shape, 2.0))
        self.assertEqual(constant.normalized_target_energy, 8.0)  # Sum over two axes.

    def test_predictions_project_independently_negative_scores_and_one_component(self):
        d = design()
        y = np.array([[[1., 0.], [1., 0.]], [[3., 0.], [3., 0.]],
                      [[8., 0.], [8., 0.]]])
        # Zero predictions have zero E projection; observed means cannot enter them.
        with patch.object(cs, "_project", wraps=cs._project) as calls:
            result = scored(d, component="E", observed=y, correct=-y)
        self.assertEqual([call.args[2] for call in calls.call_args_list], ["E"] * 4)
        self.assertEqual([rule.score for rule in result.rules], [-3., 0., 0.])
        self.assertEqual(result.correct_minus_pooled, -3.)
        self.assertEqual(result.correct_minus_swapped, -3.)
        self.assertEqual(result.regional_gain, -3.)
        # E removes the evaluation cell mean, not a training template (here zero).
        e = cs.project_component(d, bound(d, y), component="E")
        np.testing.assert_array_equal(e[:, 0, 0], [-1., 1., 0.])
        self.assertFalse(np.array_equal(e, y))

    def test_energy_boundary_nonfinite_and_overflow_are_unscoreable(self):
        d = design()
        exact = scored(d, energy_gate=2.)
        self.assertEqual(exact.status, "scoreable")
        below = scored(d, energy_gate=np.nextafter(2., np.inf))
        self.assertEqual(below.reason, "target_energy_below_gate")
        self.assertIsNone(below.regional_gain)
        self.assertEqual(below.rules, ())
        self.assertEqual(scored(d, component="E").status, "unscoreable")
        for field in ("observed", "correct", "pooled", "swapped"):
            value = np.ones(d.shape)
            value[0, 0, 0] = np.nan
            result = scored(d, **{field: value})
            self.assertEqual(result.reason, f"nonfinite_{field}")
            self.assertIsNone(result.regional_gain)
        overflow = scored(d, observed=np.full(d.shape, 1e308))
        self.assertEqual(overflow.status, "unscoreable")
        self.assertIsNone(overflow.regional_gain)

    def test_identity_and_role_refusals_precede_payload_conversion(self):
        class Forbidden:
            def __array__(self, *args, **kwargs):
                raise AssertionError("values were decoded before identity/role refusal")
        d = design()
        for altered in (replace(d, session_id="other"), replace(d, trial_ids=("r2", "r1", "l1")),
                        replace(d, time_ids=(6, 5)), replace(d, axis_ids=("new1", "new2"))):
            with self.assertRaises(cs.ComponentScoringError):
                cs.score_component(d, region="M1", component="C", energy_gate=0.01,
                                   observed=bound(d, Forbidden()), correct=bound(d, Forbidden()),
                                   pooled=bound(d, Forbidden()), swapped=bound(altered, Forbidden()))
        for role in ("training", "calibration"):
            altered = replace(d, role=role)
            value = bound(altered, Forbidden())
            with self.assertRaises(cs.ComponentScoringError):
                cs.score_component(altered, region="M1", component="C", energy_gate=0.01,
                                   observed=value, correct=value, pooled=value, swapped=value)

    def test_no_missing_support_repair_gate_default_or_shape_coercion(self):
        for change in (dict(directions=("right", "left", "up")),
                       dict(trial_ids=("r1", "r1", "l1")), dict(time_ids=()),
                       dict(trial_directions=("right", "right", "up"))):
            with self.assertRaises(cs.ComponentScoringError):
                design(**change)
        d = design()
        for gate in (0, -1, np.inf, np.nan, True):
            with self.assertRaises(cs.ComponentScoringError):
                scored(d, energy_gate=gate)
        with self.assertRaises(cs.ComponentScoringError):
            scored(d, observed=np.ones((3, 4)))
        with self.assertRaises(cs.ComponentScoringError):
            scored(d, observed=np.full(d.shape, 1+2j))
        with self.assertRaises(cs.ComponentScoringError):
            scored(d, component="all")

    def test_registered_regional_decisions_and_pairing(self):
        d = design()
        def pair(a, b):
            # Scalar arithmetic inputs produce the desired exact gains:
            # correct=Y, pooled=swapped=(1-sqrt(g))*Y for positive g;
            # for nonpositive g, correct=(1+sqrt(-g))*Y, competitors=Y.
            def one(region, gain):
                y = np.ones(d.shape)
                if gain > 0:
                    return scored(d, region=region, observed=y,
                                  pooled=(1-np.sqrt(gain))*y, swapped=(1-np.sqrt(gain))*y)
                return scored(d, region=region, observed=y,
                              correct=(1+np.sqrt(-gain))*y, pooled=y, swapped=y)
            return cs.compare_regions(m1=one("M1", a), pmd=one("PMd", b))
        self.assertEqual(pair(1., 0.25).status, "preserved")
        self.assertEqual(pair(0.0025, 0.0025).status, "unresolved")
        mixed = pair(1., -1.)
        self.assertEqual(mixed.reason, "regional_gain_signs_disagree")
        self.assertIsNone(mixed.equal_region_gain)
        self.assertEqual(pair(-1., -4.).status, "failed")
        zero_negative = pair(0., -1.)
        self.assertEqual(zero_negative.status, "failed")
        self.assertIsNone(zero_negative.equal_region_gain)
        unscoreable = cs.compare_regions(m1=scored(d, energy_gate=3.), pmd=scored(d, region="PMd"))
        self.assertEqual(unscoreable.reason, "unscoreable_region")
        with self.assertRaises(cs.ComponentScoringError):
            cs.compare_regions(m1=scored(d), pmd=scored(replace(d, session_id="other"), region="PMd"))
        # Region-specific frozen axes are allowed, but dimensions still match.
        other_axes = replace(d, axis_ids=("PMd_axis_1", "PMd_axis_2"))
        self.assertEqual(cs.compare_regions(m1=scored(d), pmd=scored(other_axes, region="PMd")).status,
                         "preserved")


if __name__ == "__main__":
    unittest.main()
