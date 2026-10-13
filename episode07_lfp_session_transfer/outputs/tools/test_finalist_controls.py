"""Three bounded control-orchestration checks; run only in coordinated Slurm.

The native packs and prediction panels are fabricated pure fixtures. Numerical
fitting is mocked here because primary_control's separate four methods test
the real refit mathematics. These methods test intervention selection, dispatch,
original/donor integrity and complete-panel composition, not scientific recovery
or eligibility. No imports, payloads or tests ran while authoring this file.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import finalist_controls as core
from fit_mismatch import derange_fit_lfp
from native_trials import (EARLIER, _snapshot, adapt_records, fit_calibration_mean,
                           predict_calibration_mean)
from test_candidate_predictions import pair
from test_fit_mismatch import fixture
from test_native_trials import records_for


def check():
    pass


def plan(number, candidate="starting_00", stream="history_assisted"):
    return core.plan_control(core.resolve_finalist(candidate, stream=stream), number)


def example_cell(member, number, key, *, shift=0.):
    prediction, baseline = pair(key)
    prediction = replace(prediction, values=_snapshot(prediction.values + shift))
    diagnostic = core.ControlDiagnostics(member.member.candidate_id, member.action,
        member.member.recipe, member.member.weight_recipe, member.recipe,
        member.weight_recipe, member.control, None, EARLIER[key[0]], None, (), ())
    return core.MemberControlCell(member.member.candidate_id, number, prediction, baseline, diagnostic)


class FinalistControlTests(unittest.TestCase):
    def test_global_optional_category_flat_members_and_equal_mass_plans(self):
        selected = core.resolve_finalist("adaptive_07", stream="history_assisted")
        self.assertEqual(tuple(member.candidate_id for member in selected.members),
                         ("starting_13", "adaptive_02"))
        self.assertEqual(tuple(member.weight for member in selected.members), (.5, .5))
        optional = core.plan_control(selected, 10)
        self.assertEqual(optional.optional_component, "alignment")
        self.assertEqual(tuple(member.action for member in optional.members), ("refit", "reuse"))
        self.assertEqual(optional.members[0].recipe.alignment, "none")
        self.assertIsNone(optional.members[0].recipe.alignment_regularization)
        # The second member must not fall through to its active nuisance term
        # when another member selects the globally earlier alignment category.
        self.assertEqual(optional.members[1].recipe.adjustment, "explicit")
        self.assertEqual(selected.members[0].recipe.alignment, "orthogonal")
        today = plan(10, "adaptive_07", "today_only")
        self.assertEqual(today.optional_component, "nuisance")
        self.assertEqual(tuple(member.control.kind for member in today.members),
                         ("remove_nuisance", "remove_nuisance"))
        equal = core.plan_control(selected, 6)
        self.assertEqual(tuple(member.action for member in equal.members), ("reuse", "refit"))
        self.assertEqual(equal.members[1].recipe.source_masses, (1 / 3,) * 3)
        self.assertEqual(equal.members[1].recipe.calibration_mass, 1.)
        self.assertIsNone(equal.members[1].weight_recipe)
        self.assertEqual(equal.members[1].member.weight_recipe.family, "calibration_similarity")
        for number, position in ((7, 1), (8, 2), (9, 3)):
            omitted = core.plan_control(selected, number)
            self.assertEqual(tuple(member.control.source_position for member in omitted.members),
                             (position, position))
            self.assertEqual(tuple(member.recipe for member in omitted.members),
                             tuple(member.recipe for member in selected.members))
            self.assertEqual(tuple(member.action for member in plan(number, "adaptive_07", "today_only").members),
                             ("reuse", "reuse"))
        self.assertEqual(core.resolve_finalist("starting_10", stream="history_assisted").members[0]
                         .recipe.regression.max_iterations, 100000)
        with self.assertRaisesRegex(ValueError, "starting_08"):
            core.resolve_finalist("starting_08", stream="today_only")
        declaration = SimpleNamespace(primary=None, source_weighting=None,
                                     member_ids=("adaptive_03", "starting_00"), member_weights=(.5, .5))
        with patch("finalist_controls.adaptive_pair", return_value=SimpleNamespace(streams=(declaration, declaration))):
            with self.assertRaisesRegex(ValueError, "nested"):
                core.resolve_finalist("adaptive_07", stream="history_assisted")
        changed = replace(selected, members=(replace(selected.members[0], weight=.6), selected.members[1]))
        with self.assertRaises(ValueError):
            core.plan_control(changed, 1)

    def test_member_dispatch_reuse_template_source_refit_and_exact_mismatch_integrity(self):
        original = fixture()
        days = tuple(day for day in original if day.region == "M1" and day.stream == "history_assisted")
        target = days[-1]
        development = adapt_records(records_for(role="development", first_trial=2000000, electrode=10),
            view_role="development_features", animal=target.animal, target_day=target.target_day,
            region=target.region, stream=target.stream, check=check)[0]
        baseline = predict_calibration_mean(fit_calibration_mean(target, check=check), development, check=check)
        prediction = replace(baseline, values=_snapshot(baseline.values - .25))
        kwargs = dict(fit_days=days, development=development,
                      original_prediction=prediction, calibration_mean=baseline, check=check)
        with patch("finalist_controls.fit_primary_model", side_effect=AssertionError("identity refit")), \
                patch("finalist_controls.fit_primary_control", side_effect=AssertionError("identity refit")):
            reused = core.run_member_cell(plan(1), "starting_00", **kwargs)
            template = core.run_member_cell(plan(5), "starting_00", **kwargs)
        np.testing.assert_array_equal(reused.prediction.values, prediction.values)
        self.assertEqual(reused.diagnostics.display_aliases,
                         tuple(zip(EARLIER[target.animal], ("source2", "source3", "source1"))))
        np.testing.assert_array_equal(template.prediction.values, baseline.values)
        self.assertEqual(template.prediction.calibration_trial_ids, target.trial_ids)
        with self.assertRaises(ValueError):
            core.run_member_cell(plan(1), "starting_00", **dict(kwargs,
                calibration_mean=replace(baseline, values=_snapshot(baseline.values + 1))))

        removal = plan(4)
        fitted = SimpleNamespace(effective_recipe=replace(removal.members[0].recipe, source_masses=()),
                                 source_weights=None, retained_source_days=())
        with patch("finalist_controls.fit_primary_control", return_value=fitted) as fit, \
                patch("finalist_controls.predict_primary_control_development", return_value=prediction):
            result = core.run_member_cell(removal, "starting_00", **kwargs)
        self.assertEqual(fit.call_args.kwargs["control"], core.PrimaryControl("remove_history"))
        self.assertTrue(all(a is b for a, b in zip(fit.call_args.args[0], days)))
        self.assertEqual(result.diagnostics.retained_source_days, ())
        self.assertEqual(result.diagnostics.effective_recipe.source_masses, ())
        equal = plan(6, "adaptive_02")
        with patch("finalist_controls.fit_primary_model", return_value=SimpleNamespace(recipe=equal.members[0].recipe)) as fit, \
                patch("finalist_controls.predict_primary_development", return_value=prediction), \
                patch("finalist_controls.fit_weighted_primary_model", side_effect=AssertionError("removed rule refitted")):
            result = core.run_member_cell(equal, "adaptive_02", **kwargs)
        self.assertEqual(fit.call_args.kwargs["recipe"].source_masses, (1 / 3,) * 3)
        self.assertIsNone(result.diagnostics.weight_recipe)
        self.assertEqual(result.diagnostics.original_weight_recipe.family, "calibration_similarity")

        mismatch = derange_fit_lfp(original, scope="earlier", check=check)
        with patch("finalist_controls.fit_primary_model", return_value=SimpleNamespace(recipe=plan(2).members[0].recipe)) as fit, \
                patch("finalist_controls.predict_primary_development", return_value=prediction):
            result = core.run_member_cell(plan(2), "starting_00", mismatch=mismatch, **kwargs)
        selected = tuple(day for day in mismatch.packs if day.region == "M1" and day.stream == target.stream)
        self.assertTrue(all(a is b for a, b in zip(fit.call_args.args[0], selected)))
        self.assertEqual(tuple(value[0] for value in result.diagnostics.donor_trial_ids),
                         tuple(day.day for day in days))
        changed = mismatch.packs[0]
        malformed = (
            replace(mismatch, scope="calibration", seed=7007102),
            replace(mismatch, packs=(replace(changed, spikes=_snapshot(changed.spikes + 1)),) + mismatch.packs[1:]),
            replace(mismatch, packs=(replace(changed, lfp=original[0].lfp),) + mismatch.packs[1:]))
        for invalid in malformed:
            with patch("finalist_controls.fit_primary_model", side_effect=AssertionError("tampered input reached fit")):
                with self.assertRaises(ValueError):
                    core.run_member_cell(plan(2), "starting_00", mismatch=invalid, **kwargs)
        today_days = tuple(day for day in original if day.region == "M1" and day.stream == "today_only")
        today_kwargs = dict(kwargs, fit_days=today_days, development=replace(development, stream="today_only"))
        with patch("finalist_controls.fit_primary_model", side_effect=AssertionError("unaffected today refit")):
            unchanged = core.run_member_cell(plan(2, stream="today_only"), "starting_00",
                                              mismatch=mismatch, **today_kwargs)
        np.testing.assert_array_equal(unchanged.prediction.values, prediction.values)
        self.assertEqual(unchanged.diagnostics.donor_trial_ids,
                         ((target.day, target.trial_ids, target.trial_ids),))
        calibration_mismatch = derange_fit_lfp(original, scope="calibration", check=check)
        for region, stream in (("PMd", "history_assisted"), ("M1", "today_only")):
            fitting = tuple(day for day in original if (day.region, day.stream) == (region, stream))
            dev = adapt_records(records_for(role="development", region=region, first_trial=2000000, electrode=10),
                view_role="development_features", animal=target.animal, target_day=target.target_day,
                region=region, stream=stream, check=check)[0]
            mean = predict_calibration_mean(fit_calibration_mean(fitting[-1], check=check), dev, check=check)
            chosen = plan(3, stream=stream)
            with patch("finalist_controls.fit_primary_model", return_value=SimpleNamespace(recipe=chosen.members[0].recipe)) as fit, \
                    patch("finalist_controls.predict_primary_development", return_value=mean):
                cell = core.run_member_cell(chosen, "starting_00", fit_days=fitting, development=dev,
                    original_prediction=mean, calibration_mean=mean, mismatch=calibration_mismatch, check=check)
            expected = tuple(day for day in calibration_mismatch.packs
                             if (day.region, day.stream) == (region, stream))
            self.assertTrue(all(a is b for a, b in zip(fit.call_args.args[0], expected)))
            self.assertTrue(all(recipient != donor for recipient, donor in
                zip(cell.diagnostics.donor_trial_ids[-1][1], cell.diagnostics.donor_trial_ids[-1][2])))

    def test_full_panel_fixed_average_retains_native_axes_and_rejects_partial_or_changed_members(self):
        selected = plan(1, "adaptive_07")
        left = tuple(example_cell(selected.members[0], 1, key) for key in core.publication.PANEL_KEYS)
        right = tuple(example_cell(selected.members[1], 1, key, shift=2.)
                      for key in core.publication.PANEL_KEYS[::-1])
        result = core.compose_control_panel(selected, (left, right), check=check)
        self.assertEqual(result.member_ids, ("starting_13", "adaptive_02"))
        self.assertEqual(result.member_weights, (.5, .5))
        self.assertFalse(hasattr(result, "check_record"))
        for key, actual, baseline in zip(core.publication.PANEL_KEYS, result.predictions, result.calibration_means):
            expected, comparator = pair(key)
            np.testing.assert_allclose(actual.values, expected.values + 1, rtol=0, atol=2e-14)
            for name in ("directions", "bin_offsets", "unit_guide"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            self.assertEqual(actual.trial_ids, expected.trial_ids)
            self.assertEqual(actual.calibration_trial_ids, expected.calibration_trial_ids)
            np.testing.assert_array_equal(baseline.values, comparator.values)
        with self.assertRaises(ValueError):
            core.compose_control_panel(selected, (left, right[:-1]), check=check)
        changed = right[0]
        altered = (
            replace(changed, prediction=replace(changed.prediction, unit_guide=changed.prediction.unit_guide[::-1]),
                    calibration_mean=replace(changed.calibration_mean, unit_guide=changed.calibration_mean.unit_guide[::-1])),
            replace(changed, calibration_mean=replace(changed.calibration_mean,
                    values=_snapshot(changed.calibration_mean.values + 1))),
            replace(changed, prediction=replace(changed.prediction,
                    values=np.full_like(changed.prediction.values, np.nan))))
        for invalid in altered:
            with self.assertRaises(ValueError):
                core.compose_control_panel(selected, (left, (invalid,) + right[1:]), check=check)
        template_plan = plan(5, "adaptive_07")
        template_panels = []
        for member in template_plan.members:
            cells = tuple(example_cell(member, 5, key) for key in core.publication.PANEL_KEYS)
            template_panels.append(tuple(replace(cell, prediction=cell.calibration_mean) for cell in cells))
        template_panels = tuple(template_panels)
        template = core.compose_control_panel(template_plan, template_panels, check=check)
        self.assertEqual(template.member_weights, (.5, .5))
        for prediction, baseline in zip(template.predictions, template.calibration_means):
            self.assertIs(prediction, baseline)
            np.testing.assert_array_equal(prediction.values, baseline.values)
        bad_cell = replace(template_panels[0][0], prediction=replace(template_panels[0][0].prediction,
                            values=_snapshot(template_panels[0][0].prediction.values + .25)))
        with self.assertRaisesRegex(ValueError, "template member"):
            core.compose_control_panel(template_plan,
                ((bad_cell,) + template_panels[0][1:], template_panels[1]), check=check)
        def cancelled():
            raise RuntimeError("stop before composition")
        with self.assertRaisesRegex(RuntimeError, "stop before composition"):
            core.compose_control_panel(selected, (left, right), check=cancelled)


if __name__ == "__main__":
    unittest.main()
