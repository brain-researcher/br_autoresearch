"""Six causal WAY traces over invented summaries, never source samples.

Run only in the coordinator's Slurm allocation. These tests exercise the
selected state/label composition; passing does not qualify sensors, thresholds,
cue availability, physical units or the acquisition/timing bound.
"""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from classical_forecast_backend import GroupIdentity
from way_completed_return import ReturnSupport
from way_risk_stream import WayRiskStream, label_from_snapshots


IDENTITY = GroupIdentity("WAY", 1, 1, "test-stream-series-1-segment-1")
LOW = (1.,) * 8
HIGH = (3.,) * 8


def values(*active):
    return tuple(4. if path in active else 0. for path in range(8))


def support(tick=250, status="supported"):
    return ReturnSupport(status, (), tick - 250, tick)


def begin(*, status="supported"):
    stream = WayRiskStream(IDENTITY, LOW, HIGH, grid_origin_tick=0)
    packet = None if status is None else support(status=status)
    first = stream.step(250, values(), start=True, return_support=packet)
    return stream, first


def feed(stream, first, last, *active):
    """A short consecutive trace with fixed active paths, inclusive endpoints."""
    return {tick: stream.step(tick, values(*active))
            for tick in range(first, last + 1)}


def visible(snapshot):
    """Compare public evidence without confusing independent stream lineage."""
    return (snapshot.cue, snapshot.state, snapshot.decision, snapshot.pending,
            snapshot.onset, snapshot.followup, snapshot.grid_origin_tick,
            snapshot.closed, snapshot.close_reason)


class ForbiddenSummary:
    def __array__(self, *args, **kwargs):
        raise AssertionError("an end cue must not inspect current summaries")


class WayRiskStreamTests(unittest.TestCase):
    def test_prefix_snapshots_stay_fixed_when_future_confirmation_differs(self):
        left, left_start = begin()
        right, right_start = begin()
        self.assertEqual(visible(left_start), visible(right_start))
        feed(left, 251, 274)
        feed(right, 251, 274)
        left_pending = left.step(275, values(0))
        right_pending = right.step(275, values(0))
        prefix = (visible(left_start), visible(left_pending))
        self.assertEqual(visible(left_pending), visible(right_pending))
        self.assertEqual(left_pending.decision.state, "pending")
        self.assertEqual([(p.path_index, p.crossing_tick)
                          for p in left_pending.pending], [(0, 275)])
        self.assertEqual(left_pending.followup.unknown_from_tick, 275)

        feed(left, 276, 294, 0)
        feed(right, 276, 279, 0)
        feed(right, 280, 294)
        confirmed = left.step(295, values(0))
        rejected = right.step(295, values())
        self.assertEqual((confirmed.onset.crossing_tick,
                          confirmed.onset.known_tick), (275, 295))
        self.assertIsNone(rejected.onset)
        self.assertEqual(prefix, (visible(left_start), visible(left_pending)))
        self.assertEqual(visible(left_pending), visible(right_pending))
        self.assertIsNone(label_from_snapshots(left_pending, confirmed).observed_category)
        self.assertIsNone(label_from_snapshots(right_pending, rejected).observed_category)
        with self.assertRaises(FrozenInstanceError):
            left_pending.state = "at_risk"

    def test_competing_paths_choose_earliest_accepted_crossing(self):
        # The earlier path rejects; the later path remains continuously high.
        stream, first = begin()
        feed(stream, 251, 269)
        feed(stream, 270, 274, 0)
        both = feed(stream, 275, 279, 0, 1)[275]
        self.assertEqual([(p.path_index, p.crossing_tick) for p in both.pending],
                         [(0, 270), (1, 275)])
        feed(stream, 280, 294, 1)
        accepted = stream.step(295, values(1))
        self.assertEqual((accepted.onset.crossing_tick, accepted.onset.known_tick),
                         (275, 295))
        self.assertEqual(label_from_snapshots(first, accepted).observed_category, 1)

        # A later pending path must not delay an already accepted earlier one.
        stream, first = begin()
        feed(stream, 251, 269)
        feed(stream, 270, 274, 0)
        feed(stream, 275, 289, 0, 1)
        accepted = stream.step(290, values(0, 1))
        self.assertEqual((accepted.onset.crossing_tick, accepted.onset.known_tick),
                         (270, 290))
        self.assertEqual(label_from_snapshots(first, accepted).observed_category, 0)

    def test_failed_or_unknown_start_never_arms_from_later_quietness(self):
        for status, initial_state in (("not_supported", "unarmed"),
                                      ("unknown", "unknown"), (None, "unknown")):
            with self.subTest(status=status):
                stream, first = begin(status=status)
                self.assertEqual(first.state, initial_state)
                trace = feed(stream, 251, 800)
                for snapshot in trace.values():
                    self.assertNotEqual(snapshot.state, "at_risk")
                    if snapshot.decision is not None:
                        self.assertNotEqual(snapshot.decision.state, "at_risk")
                self.assertIsNone(label_from_snapshots(first, trace[800]).observed_category)

        # A contradictory support interval is refused before consuming tick250.
        stream = WayRiskStream(IDENTITY, LOW, HIGH, grid_origin_tick=0)
        with self.assertRaises(ValueError):
            stream.step(250, values(), start=True,
                        return_support=ReturnSupport("supported", (), 1, 250))
        self.assertEqual(stream.step(250, values(), start=True,
                                    return_support=support()).state, "at_risk")

    def test_rejection_requalifies_exactly_and_keeps_detectors_running(self):
        for high_at_rearm in (False, True):
            with self.subTest(high_at_rearm=high_at_rearm):
                stream, first = begin()
                feed(stream, 251, 269)
                feed(stream, 270, 274, 0)
                rejected = stream.step(275, values())
                self.assertEqual(rejected.decision.state, "pending")
                trace = feed(stream, 276, 524)
                for snapshot in trace.values():
                    if snapshot.decision is not None:
                        self.assertNotEqual(snapshot.decision.state, "at_risk")
                # The fresh quiet window is [275,525), including rejection275.
                rearm = stream.step(525, values(1) if high_at_rearm else values())
                self.assertEqual(rearm.decision.state,
                                 "pending" if high_at_rearm else "at_risk")
                if high_at_rearm:
                    self.assertEqual([(p.path_index, p.crossing_tick)
                                      for p in rearm.pending], [(1, 525)])
                self.assertEqual(rejected.decision.state, "pending")

        # A new movement during requalification still labels an earlier risk row.
        stream, first = begin()
        feed(stream, 251, 269)
        feed(stream, 270, 274, 0)
        stream.step(275, values())
        feed(stream, 276, 299)
        pending = stream.step(300, values(2))
        feed(stream, 301, 319, 2)
        accepted = stream.step(320, values(2))
        self.assertEqual(pending.decision.state, "pending")
        self.assertEqual((accepted.onset.crossing_tick, accepted.onset.known_tick),
                         (300, 320))
        self.assertEqual(label_from_snapshots(first, accepted).observed_category, 2)
        self.assertIsNone(label_from_snapshots(pending, accepted).observed_category)

    def test_one_reach_needs_end_and_boundaries_precede_confirmation(self):
        stream, first = begin()
        feed(stream, 251, 269)
        feed(stream, 270, 290, 0)
        trace = feed(stream, 291, 599)
        self.assertTrue(all(snapshot.state != "at_risk" for snapshot in trace.values()))
        with self.assertRaises(ValueError):
            stream.step(600, values(), start=True, return_support=support(600))
        ended = stream.step(600, ForbiddenSummary(), end=True)
        self.assertIsNone(ended.decision)
        self.assertEqual(ended.followup.observed_stop_tick, 600)
        # An observed end at600 cannot support a claimed return [351,601).
        with self.assertRaises(ValueError):
            stream.step(601, values(), start=True, return_support=support(601))
        feed(stream, 601, 899)
        next_cue = stream.step(900, values(), start=True, return_support=support(900))
        self.assertEqual(next_cue.state, "at_risk")
        self.assertEqual(next_cue.cue.start_tick, 900)
        with self.assertRaises(ValueError):
            label_from_snapshots(first, next_cue)

        for boundary in ("end", "gap", "missing", "administrative"):
            with self.subTest(boundary=boundary):
                stream, first = begin()
                feed(stream, 251, 269)
                feed(stream, 270, 289, 0)
                if boundary == "end":
                    stopped = stream.step(290, ForbiddenSummary(), end=True)
                elif boundary == "gap":
                    stopped = stream.step(290, values(0), end=True, history_valid=False)
                    self.assertTrue(stopped.closed)
                    self.assertEqual(stopped.close_reason, "gap")
                elif boundary == "missing":
                    stopped = stream.step(290, (float("nan"),) + (0.,) * 7)
                    self.assertTrue(stopped.closed)
                    self.assertEqual(stopped.close_reason, "gap")
                else:
                    stopped = stream.close(290, reason="administrative")
                    self.assertTrue(stopped.closed)
                self.assertIsNone(stopped.decision)
                self.assertIsNone(stopped.onset)
                self.assertEqual(stopped.followup.observed_stop_tick, 290)
                self.assertEqual(stopped.followup.unknown_from_tick, 270)
                censored = label_from_snapshots(first, stopped)
                self.assertEqual((censored.observed_category, censored.complete_bins), (-1, 0))
                if boundary != "end":
                    with self.assertRaises(ValueError):
                        stream.step(291, values())

    def test_snapshot_labels_use_available_evidence_and_separate_coverage(self):
        stream, first = begin()
        feed(stream, 251, 274)
        pending = stream.step(275, values(0))
        before_confirmation = feed(stream, 276, 294, 0)[294]
        censored = label_from_snapshots(first, before_confirmation)
        self.assertEqual((censored.observed_category, censored.complete_bins,
                          censored.observed_through_bins), (-1, 1, 1))
        accepted = stream.step(295, values(0))
        event = label_from_snapshots(first, accepted)
        self.assertEqual((event.observed_category, event.complete_bins,
                          event.observed_through_bins), (1, 1, 1))
        # At native500Hz the full900ms horizon ends at exclusive tick700.
        complete = feed(stream, 296, 699)[699]
        event_later = label_from_snapshots(first, complete)
        self.assertEqual((event_later.observed_category, event_later.complete_bins,
                          event_later.observed_through_bins), (1, 1, 18))
        self.assertEqual(label_from_snapshots(first, before_confirmation), censored)
        excluded = label_from_snapshots(pending, complete)
        self.assertEqual((excluded.observed_category, excluded.complete_bins,
                          excluded.observed_through_bins), (None, None, None))
        with self.assertRaises(ValueError):
            label_from_snapshots(pending, first)
        other, other_first = begin()
        other_evidence = feed(other, 251, 300)[300]
        with self.assertRaises(ValueError):
            label_from_snapshots(first, other_evidence)


if __name__ == "__main__":
    unittest.main()
