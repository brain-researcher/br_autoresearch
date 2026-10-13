"""Three scalar-record accounting checks; run only in coordinated compute.

These fabricated feedback records test stopping and contact bookkeeping, not
model quality, scientific evidence, or provenance of actual feedback reports.
"""
from copy import deepcopy
import unittest

import adaptive_progress as core


def stream(score):
    return {"complete": True, "global_selection_r2": score,
        "regions": [{"region": region, "available": True, "unavailable_reason": None,
                     "complete_new_days": 5, "independent_animals": 2,
                     "model_r2": score, "calibration_mean_r2": 0.,
                     "improvement_over_mean": score} for region in ("M1", "PMd")]}


def pair(name, history=0., today=0.):
    return {"candidate_id": name, "outcome_contact": True, "state": "paired_development_scored",
            "streams": {key: stream(value) for key, value in zip(core.STREAMS, (history, today))}}


def failure(name):
    return {"candidate_id": name, "outcome_contact": False,
            "state": "paired_prediction_incomplete_unscored", "streams": {}}


def starting():
    # All sixteen attempted identities remain visible; the adopted exclusion
    # is unscored. Its exhausted-attempt provenance belongs to the caller.
    return [failure(name) if name == "starting_08" else pair(name)
            for name in core.STARTING_IDS]


class AdaptiveProgressTests(unittest.TestCase):
    def test_strict_global_incumbents_and_nonaccumulating_patience(self):
        starts = starting()
        records = [pair("adaptive_a", .001), pair("adaptive_b", .002), pair("adaptive_c", .004)]
        first = core.assess_adaptive_progress(starts, records[:1])
        second = core.assess_adaptive_progress(starts, records[:2])
        reset = core.assess_adaptive_progress(starts, records)
        self.assertEqual(first.incumbents[0].candidate_id, "adaptive_a")
        self.assertEqual(core.MIN_VALID, 3)
        self.assertTrue(first.starting_complete)
        self.assertFalse(first.all_attempted_starts_available)
        self.assertEqual(first.excluded_starting_ids, ("starting_08",))
        self.assertEqual(first.unscored_starting_ids, ("starting_08",))
        self.assertEqual(first.starting_deficit_ids, ())
        self.assertEqual(tuple(value.candidate_id for value in first.starting_records), core.STARTING_IDS)
        self.assertEqual(second.incumbents[0].candidate_id, "adaptive_b")
        self.assertEqual(second.consecutive_valid_without_qualifying_improvement, 2)
        self.assertEqual(reset.consecutive_valid_without_qualifying_improvement, 0)
        self.assertEqual(reset.incumbents[1].candidate_id, "starting_00")
        # The common objective is the mean of both regions; no regional winner.
        candidate = pair("adaptive_d", 0., .003)
        candidate["streams"][core.STREAMS[0]]["regions"][0].update(
            model_r2=-1., improvement_over_mean=-1.)
        candidate["streams"][core.STREAMS[0]]["regions"][1].update(
            model_r2=1., improvement_over_mean=1.)
        records.append(candidate)
        result = core.assess_adaptive_progress(starts, records)
        self.assertEqual(tuple(value.candidate_id for value in result.incumbents),
                         ("adaptive_c", "adaptive_d"))
        self.assertEqual(result.consecutive_valid_without_qualifying_improvement, 0)
        records.extend(pair(f"no_gain_{index}", -.1, -.1) for index in range(7))
        result = core.assess_adaptive_progress(starts, records)
        self.assertEqual((result.status, result.stop_reason), ("adaptive_complete", "patience_exhausted"))
        self.assertEqual(result.consecutive_valid_without_qualifying_improvement, 7)
        self.assertEqual(result.recorded_started_evaluations, (26, 26))
        self.assertEqual(result.recorded_completed_evaluations, (26, 26))
        self.assertEqual(result.paired_opportunities_per_stream, (27, 27))
        self.assertEqual(result.declared_pair_contacts, 26)
        self.assertFalse(result.whole_search_complete)
        self.assertFalse(result.authorizes_held_out)
        with self.assertRaises(ValueError):
            core.assess_adaptive_progress(starts, records + [pair("too_late")])

    def test_minimum_valid_exhaustion_and_failed_attempt_cap(self):
        starts = starting()
        for count in (0, 1, 2, 3):
            records = [pair(f"valid_{index}", .01 * (index + 1)) for index in range(count)]
            result = core.assess_adaptive_progress(starts, records, model_space_exhausted=True)
            self.assertEqual(result.adaptive_stage_complete, count == 3)
            self.assertEqual(result.status, "adaptive_complete" if count == 3 else "incomplete_minimum_valid")
        records = [failure(f"failed_{index}") for index in range(11)]
        progress = core.assess_adaptive_progress(starts, records)
        self.assertEqual((progress.valid_successor_pairs,
                          progress.consecutive_valid_without_qualifying_improvement), (0, 0))
        records += [pair("valid_a"), pair("valid_b"), pair("valid_c")]
        result = core.assess_adaptive_progress(starts, records)
        self.assertEqual((result.status, result.stop_reason), ("adaptive_complete", "maximum_attempted_pairs"))
        self.assertEqual((result.attempted_successor_pairs, result.valid_successor_pairs), (14, 3))
        self.assertEqual(result.paired_opportunities_per_stream, (30, 30))
        self.assertEqual(result.recorded_completed_evaluations, (18, 18))
        self.assertEqual(result.consecutive_valid_without_qualifying_improvement, 3)
        short = core.assess_adaptive_progress(starts, records[:-1] + [failure("last_failed")])
        self.assertEqual((short.status, short.stop_reason),
                         ("incomplete_minimum_valid", "maximum_attempted_pairs"))
        self.assertFalse(short.adaptive_stage_complete)
        all_failed = core.assess_adaptive_progress(starts, [failure(f"bad_{index}") for index in range(14)])
        self.assertEqual(all_failed.status, "incomplete_minimum_valid")
        self.assertFalse(all_failed.adaptive_stage_complete)
        with self.assertRaises(ValueError):
            core.assess_adaptive_progress(starts, records + [failure("over_cap")])
        with self.assertRaises(ValueError):
            core.assess_adaptive_progress(starts, model_space_exhausted=1)

    def test_starting_repair_merge_partial_contacts_and_duplicate_refusal(self):
        starts = starting()
        merged = starts[:]
        merged[8], merged[10] = failure("starting_08"), failure("starting_10")
        incomplete = core.assess_adaptive_progress(merged)
        self.assertEqual(incomplete.status, "continue_with_incomplete_starting")
        self.assertFalse(incomplete.starting_complete)
        self.assertEqual(incomplete.starting_deficit_ids, ("starting_10",))
        self.assertEqual(incomplete.excluded_starting_ids, ("starting_08",))
        self.assertEqual(incomplete.unscored_starting_ids, ("starting_08", "starting_10"))
        self.assertEqual(incomplete.recorded_completed_evaluations, (14, 14))
        failed_repair = core.assess_adaptive_progress(merged +
            [failure("starting_08"), failure("starting_10")])
        self.assertEqual(failed_repair.status, "continue_with_incomplete_starting")
        self.assertEqual(failed_repair.recorded_started_evaluations, (14, 14))
        self.assertEqual(failed_repair.paired_opportunities_per_stream, (16, 16))
        continued = core.assess_adaptive_progress(merged,
            [pair("new_a", .01), pair("new_b", .02), pair("new_c", .03)], model_space_exhausted=True)
        self.assertTrue(continued.adaptive_stage_complete)
        self.assertEqual(continued.status, "adaptive_complete_with_incomplete_starting")
        self.assertFalse(continued.starting_complete)
        self.assertEqual(continued.starting_deficit_ids, ("starting_10",))
        self.assertEqual(continued.recorded_completed_evaluations, (17, 17))
        self.assertFalse(continued.whole_search_complete)
        self.assertFalse(continued.authorizes_held_out)
        one_valid = [failure(name) for name in core.STARTING_IDS]
        one_valid[7] = pair("starting_07", .03, -.04)
        initialized = core.assess_adaptive_progress(one_valid, [pair("after_one", .02, -.05)])
        self.assertEqual(tuple(value.candidate_id for value in initialized.incumbents),
                         ("starting_07", "starting_07"))
        self.assertEqual(len(initialized.starting_deficit_ids), 14)
        self.assertNotIn("starting_08", initialized.starting_deficit_ids)
        historical = deepcopy(merged)
        result = core.assess_adaptive_progress(merged + [failure("starting_08"), starts[10]])
        self.assertEqual(merged, historical)
        self.assertTrue(result.starting_complete)
        self.assertFalse(result.all_attempted_starts_available)
        self.assertEqual(result.unscored_starting_ids, ("starting_08",))
        self.assertEqual(result.excluded_starting_ids, ("starting_08",))
        self.assertEqual(result.starting_deficit_ids, ())
        self.assertEqual(result.recorded_started_evaluations, (15, 15))
        self.assertEqual(result.declared_pair_contacts, 15)
        self.assertEqual(result.paired_opportunities_per_stream, (16, 16))
        partial = pair("partial")
        partial["state"] = "paired_development_contact_started"
        partial["streams"]["today_only"] = {"state": "development_contact_started"}
        result = core.assess_adaptive_progress(starts, [partial])
        self.assertEqual(result.status, "incomplete_technical")
        self.assertEqual(result.incomplete_contact_ids, ("partial",))
        self.assertEqual(result.recorded_started_evaluations, (16, 16))
        self.assertEqual(result.recorded_completed_evaluations, (16, 15))
        self.assertEqual(result.valid_successor_pairs, 0)
        self.assertEqual(tuple(value.candidate_id for value in result.incumbents),
                         ("starting_00", "starting_00"))
        unknown = deepcopy(partial)
        unknown["streams"] = {}
        result = core.assess_adaptive_progress(starts, [unknown])
        self.assertEqual(result.unattributed_declared_pair_contacts, 1)
        self.assertEqual(result.recorded_started_evaluations, (15, 15))
        unavailable = pair("unavailable")
        bad = unavailable["streams"]["today_only"]
        bad["complete"], bad["global_selection_r2"] = False, None
        bad["regions"][0].update(available=False, unavailable_reason="zero_variance",
                                 model_r2=None, calibration_mean_r2=None, improvement_over_mean=None)
        result = core.assess_adaptive_progress(starts, [unavailable])
        self.assertEqual(result.status, "incomplete_technical")
        self.assertEqual(result.recorded_completed_evaluations, (16, 16))
        self.assertEqual(result.available_evaluations, (16, 15))
        # No successful, unavailable or partial contact for the excluded ID
        # can be merged over the historical unscored attempt.
        for contacted in (pair("starting_08"), partial, unavailable):
            contacted = deepcopy(contacted)
            contacted["candidate_id"] = "starting_08"
            with self.assertRaisesRegex(ValueError, "excluded starting_08"):
                core.assess_adaptive_progress(starts + [contacted])
        # Individually available scores from an unpaired/unavailable start do
        # not nominate an incumbent, even when much larger than valid pairs.
        for bad_start in (partial, unavailable):
            bad_start = deepcopy(bad_start)
            bad_start["candidate_id"] = "starting_00"
            bad_start["streams"]["history_assisted"] = stream(999.)
            blocked = [bad_start] + starts[1:]
            result = core.assess_adaptive_progress(blocked)
            self.assertEqual(result.status, "incomplete_technical")
            self.assertEqual(tuple(value.candidate_id for value in result.incumbents),
                             ("starting_01", "starting_01"))
            with self.assertRaises(ValueError):
                core.assess_adaptive_progress(blocked, [pair("cannot_advance_partial")])
        none_valid = [failure(name) for name in core.STARTING_IDS]
        self.assertEqual(core.assess_adaptive_progress(none_valid).status, "incomplete_starting")
        for initial, successors in ((starts + [starts[0]], ()), (starts[:-1], ()),
                (none_valid, [pair("too_early")]), (starts, [failure("again"), pair("again")]),
                (starts, [pair("starting_00")])):
            with self.assertRaises(ValueError):
                core.assess_adaptive_progress(initial, successors)
        inconsistent = pair("bad_objective")
        inconsistent["streams"]["history_assisted"]["global_selection_r2"] = 1.
        with self.assertRaises(ValueError):
            core.assess_adaptive_progress(starts, [inconsistent])


if __name__ == "__main__":
    unittest.main()
