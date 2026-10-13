"""Two fixed-construction/oracle checks; execute only in allocated compute.

No finalist is fitted by these tests. They verify the prospectively selected
integer construction and independent scalar scientific oracles, not a second
known-answer campaign or an empirical result.
"""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import known_answer_cases as core
from finalist_controls import resolve_finalist
from native_trials import BANDS, EARLIER, NativePredictions, _snapshot


def check():
    pass


def predicted(cell, values):
    day, target = cell.development, cell.fit_days[-1]
    full = np.broadcast_to(values, cell.evaluation_counts.shape)
    return NativePredictions(day.animal, day.day, day.region, day.trial_ids,
        day.directions, day.bin_offsets, target.unit_guide, _snapshot(full), target.trial_ids)


class KnownAnswerCaseTests(unittest.TestCase):
    def test_fixed_integer_support_conditional_balance_and_native_signal_geometry(self):
        before = np.random.get_state()
        for name in ("stable", "drift", "null", "scale"):
            generated = core.generate_case(name, check=check)
            self.assertEqual(tuple((cell.region, cell.stream) for cell in generated.cells),
                (("M1", "history_assisted"), ("M1", "today_only"),
                 ("PMd", "history_assisted"), ("PMd", "today_only")))
            scale = 2 if name == "scale" else 1
            shared_ids = {}
            for cell in generated.cells:
                development = cell.development
                self.assertEqual((development.view_role, development.role),
                                 ("development_features", "development"))
                self.assertIsNone(development.spikes)
                self.assertIsNone(development.unit_guide)
                self.assertEqual(len(development.trial_ids), 256)
                np.testing.assert_array_equal(np.bincount(development.directions), np.full(8, 32))
                self.assertFalse(set(development.trial_ids).intersection(cell.fit_days[-1].trial_ids))
                self.assertEqual(cell.evaluation_counts.shape, (256, 10 if cell.region == "M1" else 15, 16))
                self.assertTrue(np.issubdtype(cell.evaluation_counts.dtype, np.integer))
                self.assertGreaterEqual(int(cell.evaluation_counts.min()), 0)
                evaluation_index = np.array([int(value[6:]) - 1001 for value in development.trial_ids])
                np.testing.assert_array_equal(development.directions, evaluation_index // 32)
                np.testing.assert_array_equal(cell.x, 1 - 2 * (evaluation_index % 2))
                expected_h = 1 - 2 * ((development.directions[:, None]
                                      + np.arange(len(development.bin_offsets))) % 2)
                np.testing.assert_array_equal(cell.h, expected_h)
                evaluation_beta = 0 if name == "null" else -8 if name == "drift" else 8
                evaluation_noise = cell.evaluation_counts - (64 + scale * (
                    4 * expected_h[:, :, None] + evaluation_beta * cell.x[:, None, None]))
                for direction in range(8):
                    for sign in (-1, 1):
                        selected = (development.directions == direction) & (cell.x == sign)
                        np.testing.assert_array_equal(evaluation_noise[selected].sum(axis=0),
                            np.zeros((len(development.bin_offsets), 16), dtype=int))
                expected_days = EARLIER["Mihili"] + ("20140304",) if cell.stream == "history_assisted" else ("20140304",)
                self.assertEqual(tuple(day.day for day in cell.fit_days), expected_days)
                for day in cell.fit_days:
                    self.assertEqual(day.spikes.shape[0], 128)
                    np.testing.assert_array_equal(np.bincount(day.directions), np.full(8, 16))
                    if day.day in shared_ids:
                        self.assertEqual(day.trial_ids, shared_ids[day.day])
                    shared_ids[day.day] = day.trial_ids
                    offsets = np.arange(5, 15) if day.region == "M1" else np.arange(-15, 0)
                    np.testing.assert_array_equal(day.bin_offsets, offsets)
                    numeric = np.array([int(value[6:]) - 1 for value in day.trial_ids])
                    np.testing.assert_array_equal(day.directions, numeric // 16)
                    t = numeric % 16
                    x = 1 - 2 * (t % 2)
                    h = 1 - 2 * ((day.directions[:, None] + np.arange(len(offsets))) % 2)
                    beta = 0 if name == "null" else -8 if name == "drift" and day.role == "calibration" else 8
                    conditional = 64 + scale * (4 * h + beta * x[:, None])
                    residual = day.spikes - conditional[:, :, None]
                    np.testing.assert_array_equal(residual.sum(axis=2), np.zeros(h.shape, dtype=int))
                    for direction in range(8):
                        for sign in (-1, 1):
                            rows = (day.directions == direction) & (x == sign)
                            self.assertEqual(int(rows.sum()), 8)
                            np.testing.assert_array_equal(residual[rows].sum(axis=0),
                                                          np.zeros((len(offsets), 16), dtype=int))
                    # Independently recover all fifteen contrast signs on one
                    # native row, undoing only this day's stored column order.
                    row = int(np.flatnonzero(t == 0)[0])
                    native_order = np.argsort(day.unit_guide[:, 0])
                    for bin_index in (0, len(offsets) - 1):
                        signs = [1] * 14 + [int(h[row, bin_index])]
                        expected_noise = np.array(signs + [-sum(signs)]) * scale
                        np.testing.assert_array_equal(residual[row, bin_index, native_order], expected_noise)
                    # Known native covariance eigenvalues: fourteen unit
                    # contrasts, one eigenvalue16, and the common signal.
                    centered = day.spikes.reshape(-1, 16) - 64
                    covariance = centered.T @ centered / len(centered)
                    expected_eigenvalues = np.array([1.] * 14 + [16., 16. * (16 + beta**2)]) * scale**2
                    np.testing.assert_allclose(np.linalg.eigvalsh(covariance), expected_eigenvalues,
                                               rtol=0, atol=1e-9)
                    # Numeric-ID alternating halves both have the same
                    # centered trajectory: the fixed reliability oracle is1.
                    halves = []
                    for half in (0, 1):
                        means = []
                        for direction in range(8):
                            order = sorted(np.flatnonzero(day.directions == direction), key=lambda i: numeric[i])
                            means.append(day.spikes[order[half::2]].mean(axis=0))
                        values = np.stack(means).reshape(-1, 16)
                        halves.append(values - values.mean(axis=0))
                    np.testing.assert_array_equal(halves[0], halves[1])
                    self.assertGreater(float(np.sum(halves[0]**2)), 0.)
                    for band_index, band in enumerate(BANDS):
                        columns = np.flatnonzero(np.all(day.lfp_guide[:, 1:] == band, axis=1))
                        self.assertEqual(len(columns), 16)
                        sign = x[:, None] if band_index % 2 == 0 else h
                        np.testing.assert_array_equal(day.lfp[:, :, columns].mean(axis=2),
                            np.broadcast_to(sign / 4, h.shape))
                for values in (cell.x, cell.h, cell.evaluation_counts, development.lfp):
                    with self.assertRaises(ValueError):
                        values.setflags(write=True)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        def cancelled():
            raise RuntimeError("cancel before construction")
        with patch.object(core, "_arrays", side_effect=AssertionError("construction started")):
            with self.assertRaisesRegex(RuntimeError, "cancel before construction"):
                core.generate_case("stable", check=cancelled)

    def test_independent_pooled_and_ensemble_oracles_recovery_gates_and_recipe_admission(self):
        finalists = tuple(resolve_finalist("adaptive_07", stream=stream) for stream in core.STREAMS)
        records = core.qualify_finalists(finalists)
        self.assertEqual(len(records), 8)
        history = finalists[0]
        self.assertEqual(core.member_oracle(history.members[0], "drift", stream=history.stream), (4., 4.))
        self.assertEqual(core.member_oracle(history.members[1], "drift", stream=history.stream), (4., 0.))
        # Independent scalar penalty bound for NN32(.01)+ridge(.003), whose
        # drift oracles have slopes4 and0; the ensemble's slope is2.
        neural_penalty = .02 * (16 / (math.sqrt(10) / 2) + 16 / (math.sqrt(8) / 2))
        ridge_penalty = .003 * 128 / 2
        expected_bound = (.5 * math.sqrt(2 * neural_penalty / 16)
                          + .5 * math.sqrt(2 * ridge_penalty / 16))**2 / 20
        self.assertAlmostEqual(next(value for stream, case, value in records
                                   if stream == "history_assisted" and case == "drift"), expected_bound, places=14)
        for case in core.CASES:
            generated = core.generate_case(case, check=check)
            for cell in generated.cells:
                finalist = finalists[core.STREAMS.index(cell.stream)]
                amplitude = 8. if case == "scale" else 4.
                beta = (16. if case == "scale" else 0. if case == "null" else
                        (2. if cell.stream == "history_assisted" else -8.) if case == "drift" else 8.)
                expected = 64 + amplitude * cell.h[:, :, None] + beta * cell.x[:, None, None]
                prediction = predicted(cell, expected)
                result = core.evaluate_prediction(cell, prediction, case=case, finalist=finalist)
                self.assertTrue(result.passed)
                self.assertEqual((result.oracle_beta, result.prediction_beta, result.nrmse), (beta, beta, 0.))
                if case == "stable":
                    for inadequate in (np.full_like(expected, 64.), 64 + 4 * cell.h[:, :, None]):
                        result = core.evaluate_prediction(cell, predicted(cell, inadequate), case=case, finalist=finalist)
                        self.assertFalse(result.passed)
                    # Equal/opposite native errors cannot disappear through
                    # averaging the population signal before computing MSE.
                    wrong_axis = prediction.values.copy()
                    wrong_axis[:, :, 0] += 100
                    wrong_axis[:, :, 1] -= 100
                    result = core.evaluate_prediction(cell, replace(prediction, values=_snapshot(wrong_axis)),
                                                      case=case, finalist=finalist)
                    self.assertAlmostEqual(result.prediction_beta, beta)
                    self.assertFalse(result.passed)
                if case == "null":
                    leaky = predicted(cell, expected + .3 * cell.x[:, None, None])
                    result = core.evaluate_prediction(cell, leaky, case=case, finalist=finalist)
                    self.assertLess(result.nrmse, .20)
                    self.assertGreater(result.slope_error, result.slope_limit)
                    self.assertFalse(result.passed)
                with self.assertRaises(ValueError):
                    core.evaluate_prediction(cell, replace(prediction, unit_guide=_snapshot(prediction.unit_guide[::-1])),
                                             case=case, finalist=finalist)
        # No construction or fit may start for a family/bias outside the
        # prospectively supported finalists; never lower its penalty here.
        with patch.object(core, "generate_case", side_effect=AssertionError("generated unsupported case")):
            for candidate in ("adaptive_06", "starting_03"):
                unsupported = tuple(resolve_finalist(candidate, stream=stream) for stream in core.STREAMS)
                with self.assertRaises(core.UnsupportedKnownAnswerRecipe):
                    core.run_known_answer_cases(unsupported, check=check)
        forged = replace(history, members=(replace(history.members[0], weight=.6), history.members[1]))
        with self.assertRaises(ValueError):
            core.qualify_finalists((forged, finalists[1]))
        # Equal masses alone do not prove that learned reliability/distances
        # recovered their known values. These mocks exercise diagnostics only.
        for candidate, fields in (("adaptive_09", ("reliability", "shrunk_reliability")),
                                  ("adaptive_02", ("prototype_distances",))):
            member = resolve_finalist(candidate, stream="history_assisted").members[0]
            components = np.column_stack((np.full(16, .25), np.zeros((16, 3))))
            primary = SimpleNamespace(population=SimpleNamespace(
                bases=(SimpleNamespace(components=components),), source_maps=(np.eye(4),)))
            weights = SimpleNamespace(relative_weights=np.full(3, 1 / 3),
                reliability=None, shrunk_reliability=None, prototype_distances=None)
            for field in fields:
                setattr(weights, field, np.full(3, 0. if field == "prototype_distances" else 1.))
            fitted = SimpleNamespace(primary=primary, source_weights=weights,
                effective_recipe=replace(member.recipe,
                    source_masses=(member.weight_recipe.source_total_mass / 3,) * 3,
                    calibration_mass=member.weight_recipe.target_mass))
            core._fitted_diagnostics(fitted, member, None)
            for field in fields:
                correct = getattr(weights, field)
                for invalid in (np.full(3, .5), None):
                    setattr(weights, field, invalid)
                    with self.assertRaises(core.KnownAnswerConstructionError):
                        core._fitted_diagnostics(fitted, member, None)
                setattr(weights, field, correct)


if __name__ == "__main__":
    unittest.main()
