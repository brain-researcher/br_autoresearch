"""One prospective frozen timing oracle; execute only in coordinated Slurm."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as core
import elapsed_time_baseline as timing
from development_views import DevelopmentRole
from test_bandwise_families import cca_fixture, cca_recipe
from test_bandwise_prediction import ForbiddenArray


def check():
    pass


def metadata(lfp, starts):
    start = 100 if lfp.design.role == "evaluation" else 0
    return DevelopmentRole("Mihili", lfp.region, lfp.design,
        tuple(range(start, start + len(lfp.design.trial_ids))),
        tuple(direction * math.pi / 4 for direction in lfp.design.trial_directions),
        (-1,) * len(lfp.design.trial_ids), tuple(int(value) for value in starts),
        tuple(int(value) + 10 for value in starts), tuple(range(1, 97)),
        tuple(range(15)), lfp.electrode_ids, ((1., 1.), (2., 1.)))


class ElapsedTimeBaselineTests(unittest.TestCase):
    def test_training_cell_centered_scalar_predicts_shifted_elapsed_times_on_frozen_native_axes(self):
        training, counts, u, v = cca_fixture()
        source = core.fit_bandwise(training, counts, recipe=cca_recipe(), check=check)
        directions = np.asarray(training.design.trial_directions)
        # Native start offsets vary strongly by direction, while the within-
        # direction scalar is4u+2v. Orthogonal signs give variance20 and
        # covariances(12,2) with the true residual coordinates(3u,v).
        starts = 100 + 10 * directions + 4 * u[:, 0] + 2 * v[:, 0]
        train_meta = metadata(training, starts)
        fitted = timing.fit_elapsed_time_baseline(source, train_meta, counts, check=check)
        np.testing.assert_array_equal(fitted.start_mean_by_direction, 100 + 10 * np.arange(8))
        self.assertAlmostEqual(fitted.elapsed_scale, .03 * math.sqrt(20), places=14)
        expected_coefficients = np.array((12., 2.)) / (math.sqrt(20) * 1.01)
        np.testing.assert_allclose(fitted.coefficients, expected_coefficients, rtol=0, atol=2e-13)
        np.testing.assert_allclose(fitted.intercept, np.zeros(2), rtol=0, atol=2e-13)
        self.assertIs(fitted.source_context, source)
        for value in (fitted.start_mean_by_direction, fitted.coefficients, fitted.intercept):
            with self.assertRaises(ValueError):
                value.setflags(write=True)

        evaluation, observed, shifted_u, shifted_v = cca_fixture(evaluation=True)
        evaluation_starts = (100 + 10 * directions + 4 * shifted_u[:, 0]
                             + 2 * shifted_v[:, 0] + 5)
        eval_meta = metadata(evaluation, evaluation_starts)
        predicted = timing.predict_elapsed_time_baseline(fitted, eval_meta, check=check)
        scalar = (4 * shifted_u + 2 * shifted_v + 5) / math.sqrt(20)
        expected = scalar[:, :, None] * expected_coefficients
        np.testing.assert_allclose(predicted.values, expected, rtol=0, atol=1e-11)
        self.assertEqual(predicted.design.axis_ids, source.population_axis_ids)
        self.assertEqual(predicted.design.trial_ids, evaluation.design.trial_ids)
        truth = np.stack((3 * shifted_u + 7, shifted_v + 11), axis=2)
        energy = math.fsum(float(value)**2 for value in truth.flat)
        error = math.fsum((float(y) - float(p))**2 for y, p in zip(truth.flat, expected.flat))
        scored = timing.score_elapsed_time_baseline(fitted, observed, predicted, check=check)
        self.assertAlmostEqual(scored.residual_energy, energy, places=9)
        self.assertAlmostEqual(scored.squared_error, error, places=8)
        self.assertAlmostEqual(scored.score, 1 - error / energy, places=12)
        self.assertEqual(scored.direction_time_score, 0.)
        self.assertIsNone(scored.unavailable_reason)
        # The template alone exactly predicts these native counts. Zero
        # residual energy is unavailable, not a fabricated successful score.
        template_counts = np.broadcast_to(np.array((100., 200.)), observed.design.shape).copy()
        zero = timing.score_elapsed_time_baseline(fitted,
            replace(observed, values=template_counts), predicted, check=check)
        self.assertEqual(zero.unavailable_reason, "zero_residual_energy")
        self.assertIsNone(zero.score)
        self.assertIsNone(zero.direction_time_score)

        unread = replace(counts, values=ForbiddenArray())
        with self.assertRaises(PermissionError):
            timing.fit_elapsed_time_baseline(source,
                replace(train_meta, design=replace(train_meta.design, role="evaluation")), unread, check=check)
        with self.assertRaises(ValueError):
            timing.fit_elapsed_time_baseline(source, train_meta,
                replace(unread, design=replace(unread.design, axis_ids=unread.design.axis_ids[::-1])), check=check)
        with self.assertRaises(ValueError):
            timing.fit_elapsed_time_baseline(source,
                replace(train_meta, window_stops=tuple(value + 1 for value in train_meta.window_stops)), unread, check=check)
        with self.assertRaises(ValueError):
            timing.predict_elapsed_time_baseline(fitted,
                replace(eval_meta, design=replace(eval_meta.design, axis_ids=eval_meta.design.axis_ids[::-1])), check=check)
        with self.assertRaises(PermissionError):
            timing.predict_elapsed_time_baseline(fitted, replace(eval_meta, region="PMd"), check=check)
        with self.assertRaises(PermissionError):
            timing.predict_elapsed_time_baseline(fitted, train_meta, check=check)
        def stopped():
            raise TimeoutError("caller cancellation")
        with self.assertRaises(TimeoutError):
            timing.predict_elapsed_time_baseline(fitted, eval_meta, check=stopped)


if __name__ == "__main__":
    unittest.main()
