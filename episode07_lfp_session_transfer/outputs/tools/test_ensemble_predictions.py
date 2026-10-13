"""Three bounded array-only ensemble checks; run only in allocated compute.

The complete-panel fixture is fabricated and carries fabricated evaluation
records. It tests composition/access invariants, not scientific performance.
Its shared ``pair`` fixture comes from test_candidate_predictions; neither
that module's filesystem tests nor any scorer is invoked here.
"""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import candidate_predictions as publication
import ensemble_predictions as core
from native_trials import _snapshot
from test_candidate_predictions import pair


STREAM = "history_assisted"
MEMBERS = ("starting_00", "starting_09")


def check():
    pass


def feedback(candidate_id, stream=STREAM):
    return {"candidate_id": candidate_id, "state": "paired_development_scored",
            "outcome_contact": True, "streams": {stream: {
                "complete": True, "global_selection_r2": .05,
                "regions": [{"region": region, "available": True,
                    "unavailable_reason": None, "complete_new_days": 5,
                    "independent_animals": 2, "model_r2": .05,
                    "calibration_mean_r2": .01, "improvement_over_mean": .04}
                    for region in ("M1", "PMd")]}}}


def fixture():
    pairs = tuple(pair(key) for key in publication.PANEL_KEYS)
    left = publication.CandidatePredictions(MEMBERS[0], STREAM,
        tuple(p[0] for p in pairs), tuple(p[1] for p in pairs),
        publication.PublicationCheck(publication.CHECK_KIND, 10))
    right = replace(left, candidate_id=MEMBERS[1], predictions=tuple(
        replace(prediction, values=_snapshot(baseline.values - .75
            - np.arange(len(prediction.trial_ids))[:, None, None] / 50))
        for prediction, baseline in pairs))
    return (left, right), tuple(feedback(name) for name in MEMBERS)


def compose(members, records, **kwargs):
    options = dict(candidate_id="adaptive_ensemble", stream=STREAM,
                   member_ids=MEMBERS, feedback_records=records, check=check)
    options.update(kwargs)
    return core.compose_ensemble(members, **options)


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("prediction array accessed before evaluation admission")


class EnsemblePredictionTests(unittest.TestCase):
    def test_independent_native_average_whole_panel_and_immutable_output(self):
        members, records = fixture()
        # A different cell storage order is allowed; identities determine pairing.
        reversed_member = replace(members[1], predictions=members[1].predictions[::-1],
                                  calibration_means=members[1].calibration_means[::-1])
        records[1]["streams"][STREAM]["global_selection_r2"] = -.7
        result = compose((members[0], reversed_member), records, weights=(.75, .25))
        self.assertEqual((result.candidate_id, result.stream, result.member_ids, result.weights),
                         ("adaptive_ensemble", STREAM, MEMBERS, (.75, .25)))
        self.assertFalse(hasattr(result, "check_record"))
        self.assertEqual(tuple((p.animal, p.target_day, p.region) for p in result.predictions),
                         publication.PANEL_KEYS)
        for output, baseline, original in zip(result.predictions, result.calibration_means,
                                              members[0].predictions):
            rows = np.arange(len(output.trial_ids))[:, None, None]
            # Independent simplification of .75*(base-.25+r/100)
            # + .25*(base-.75-r/50), not an average of any feedback R2.
            expected = baseline.values - .375 + rows / 400
            np.testing.assert_allclose(output.values, expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(original.values, baseline.values - .25 + rows / 100)
            self.assertEqual(output.trial_ids, original.trial_ids)
            self.assertEqual(output.calibration_trial_ids, original.calibration_trial_ids)
            np.testing.assert_array_equal(output.unit_guide, original.unit_guide)
            np.testing.assert_array_equal(output.bin_offsets, original.bin_offsets)
            self.assertEqual(output.values.shape[-1], 2 if output.region == "M1" else 3)
            self.assertLess(output.values[0, 0, 0], 0.)  # no output clipping
            publication._pair(output, baseline)  # writer-compatible native objects
            for array in (output.values, output.unit_guide, baseline.values):
                with self.assertRaises(ValueError):
                    array.setflags(write=True)
        default = compose(members, records)
        np.testing.assert_allclose(default.predictions[0].values,
            default.calibration_means[0].values - .5 - np.arange(48)[:, None, None] / 200,
            rtol=1e-14, atol=1e-14)

    def test_late_identity_or_baseline_mismatch_refused_before_any_average(self):
        members, records = fixture()
        right = members[1]
        pred, baseline = right.predictions[-1], right.calibration_means[-1]
        permuted = lambda value: replace(value, unit_guide=_snapshot(value.unit_guide[::-1]),
                                         values=_snapshot(value.values[:, :, ::-1]))
        axis_mismatch = replace(right, predictions=right.predictions[:-1] + (permuted(pred),),
            calibration_means=right.calibration_means[:-1] + (permuted(baseline),))
        baseline_mismatch = replace(right, calibration_means=right.calibration_means[:-1]
            + (replace(baseline, values=_snapshot(baseline.values + 1)),))
        new_calibration = tuple(f"trial_{401 + i:06d}" for i in range(24))
        calibration_mismatch = replace(right,
            predictions=right.predictions[:-2] + tuple(replace(value,
                calibration_trial_ids=new_calibration) for value in right.predictions[-2:]),
            calibration_means=right.calibration_means[:-2] + tuple(replace(value,
                calibration_trial_ids=new_calibration) for value in right.calibration_means[-2:]))
        incomplete = replace(right, predictions=right.predictions[:-1],
                             calibration_means=right.calibration_means[:-1])
        with patch.object(core, "_weighted_values", side_effect=AssertionError("averaging started")):
            for malformed in (axis_mismatch, baseline_mismatch, calibration_mismatch, incomplete):
                with self.subTest(malformed=malformed is incomplete):
                    with self.assertRaises(ValueError):
                        compose((members[0], malformed), records)

    def test_evaluated_same_stream_members_global_weights_and_cancellation_required(self):
        members, records = fixture()
        sentinel = replace(members[0], predictions=(replace(members[0].predictions[0],
            values=ForbiddenArray()),) + members[0].predictions[1:])
        absent = deepcopy(records)
        absent[0]["outcome_contact"] = False
        incomplete = deepcopy(records)
        incomplete[0]["streams"][STREAM]["complete"] = False
        missing_region = deepcopy(records)
        missing_region[0]["streams"][STREAM]["regions"][1]["complete_new_days"] = 4
        wrong_identity = deepcopy(records)
        wrong_identity[0]["candidate_id"] = "unevaluated"
        wrong_stream = (feedback(MEMBERS[0], "today_only"), records[1])
        for invalid in (absent, incomplete, missing_region, wrong_identity, wrong_stream, ()):
            with self.assertRaises(PermissionError):
                compose((sentinel, members[1]), invalid)
        with self.assertRaises(ValueError):
            compose((members[0], replace(members[1], stream="today_only")), records)
        with self.assertRaises(ValueError):
            compose(members, records, member_ids=MEMBERS[::-1])
        with self.assertRaises(ValueError):
            compose(members, records, candidate_id=MEMBERS[0])
        for invalid in ((.5, .4), (0., 1.), (-.1, 1.1), (True, 0.), (.3, .3, .4)):
            with self.assertRaises(ValueError):
                compose(members, records, weights=invalid)
        def cancel():
            raise RuntimeError("cancelled before composition")
        with self.assertRaisesRegex(RuntimeError, "cancelled before composition"):
            compose((sentinel, members[1]), records, check=cancel)


if __name__ == "__main__":
    unittest.main()
