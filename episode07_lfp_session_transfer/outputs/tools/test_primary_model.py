"""Small end-to-end assembly checks, not a synthetic scientific study.

Execute only in coordinated Slurm. Fabricated counts satisfy the native role
contract; targets are never provided to the development prediction function.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lfp_features import FeatureRecipe
from primary_model import PrimaryRecipe, fit_primary_model, predict_primary_development
from regression import RegressionRecipe
from test_population_bridge import full_roster


def check():
    pass


def recipe(stream, adjustment="explicit"):
    return PrimaryRecipe(FeatureRecipe(quantiles=(.5,)), 4,
        "none" if stream == "today_only" else "orthogonal", None,
        adjustment, RegressionRecipe("ridge", 0.),
        source_masses=() if stream == "today_only" else (1., 1., 1.))


def development(target):
    directions = np.repeat(np.arange(8, dtype=np.int64), 6)
    return replace(target, view_role="development_features", role="development",
        trial_ids=tuple(f"trial_{1001 + i:06d}" for i in range(48)),
        directions=directions, lfp=np.zeros((48, 10, 9)), spikes=None, unit_guide=None)


class PrimaryModelTests(unittest.TestCase):
    def test_today_complete_fit_predict_restores_primary_native_response(self):
        target = replace(full_roster()[0][-1], stream="today_only")
        dev = development(target)
        expected = np.stack([target.spikes[np.flatnonzero(target.directions == d)[0]]
                             for d in dev.directions])
        for adjustment in ("explicit", "residualize", "both"):
            with self.subTest(adjustment=adjustment):
                fitted = fit_primary_model((target,), recipe=recipe("today_only", adjustment), check=check)
                prediction = predict_primary_development(fitted, dev, check=check)
                self.assertEqual(prediction.trial_ids, dev.trial_ids)
                self.assertEqual(prediction.calibration_trial_ids, target.trial_ids)
                np.testing.assert_array_equal(prediction.unit_guide, target.unit_guide)
                np.testing.assert_allclose(prediction.values, expected, rtol=0, atol=1e-9)
                with self.assertRaises(ValueError):
                    prediction.values.setflags(write=True)

    def test_history_day_masses_not_row_counts_determine_fit_weight(self):
        days = full_roster()[0]
        chosen = replace(recipe("history_assisted"), calibration_mass=3., source_masses=(1., 2., 4.))
        from primary_model import fit_regression as actual_fit
        recorded = {}
        def fit(*args, **kwargs):
            recorded["weights"] = kwargs["row_weights"].copy()
            recorded["x_rows"] = len(args[0])
            return actual_fit(*args, **kwargs)
        with patch("primary_model.fit_regression", side_effect=fit):
            fitted = fit_primary_model(days, recipe=chosen, check=check)
        self.assertEqual(fitted.day_masses, tuple(zip((d.day for d in days), (.1, .2, .4, .3))))
        self.assertEqual(fitted.rows_per_day, tuple((d.day, len(d.trial_ids) * 10) for d in days))
        offset = 0
        for day, mass in zip(days, (.1, .2, .4, .3)):
            rows = len(day.trial_ids) * 10
            np.testing.assert_allclose(recorded["weights"][offset:offset + rows], mass / rows,
                                       rtol=1e-14, atol=0)
            offset += rows
        self.assertEqual(offset, recorded["x_rows"])
        result = predict_primary_development(fitted, development(days[-1]), check=check)
        self.assertEqual(result.values.shape, (48, 10, 7))

    def test_closed_roles_streams_and_missing_intercept_fail_before_fitting(self):
        target = replace(full_roster()[0][-1], stream="today_only")
        chosen = recipe("today_only")
        with patch("primary_model.fit_primary_population", side_effect=AssertionError("unexpected fit")):
            with self.assertRaises(PermissionError):
                fit_primary_model((development(target),), recipe=chosen, check=check)
            with self.assertRaises(ValueError):
                fit_primary_model((target,), recipe=replace(chosen, source_masses=(1., 1., 1.)), check=check)
            with self.assertRaises(ValueError):
                fit_primary_model((target,), recipe=replace(chosen,
                    regression=RegressionRecipe("ridge", 0., fit_intercept=False)), check=check)
        fitted = fit_primary_model((target,), recipe=chosen, check=check)
        dev = development(target)
        for wrong in (replace(dev, stream="history_assisted"),
                      replace(dev, view_role="held_out", role="held_out"),
                      replace(dev, spikes=np.zeros((48, 10, 7))),
                      replace(dev, trial_ids=target.trial_ids + dev.trial_ids[:24])):
            with self.assertRaises((PermissionError, ValueError)):
                predict_primary_development(fitted, wrong, check=check)


if __name__ == "__main__":
    unittest.main()
