"""Pure identity/budget checks; run inside coordinated Slurm, never on login."""
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_coverage as coverage
from plan_full_decoder_work import plan_from_spans, remaining_after


class FullDecoderPlanTests(unittest.TestCase):
    def test_every_participant_and_unit_retained_or_assigned_once(self):
        retained = tuple(coverage.retained_spans())
        plan = plan_from_spans(retained)
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["complete_decoder"])
        self.assertEqual(plan["retained_units"], 51328)
        self.assertEqual(plan["remaining_units"], 74255112)
        self.assertEqual(plan["participants"], list(coverage.PARTICIPANTS))
        self.assertTrue(75 <= len(plan["work_items"]) <= 80)
        self.assertEqual(len({item["work_item_id"] for item in plan["work_items"]}),
                         len(plan["work_items"]))
        self.assertLessEqual(plan["prospective_aggregate"]["maximum_new_durable_bytes"], 500_000_000_000)
        self.assertEqual(plan["prospective_aggregate"]["maximum_parent_jobs"], len(plan["work_items"]) + 2)
        spans = []
        for item in plan["work_items"]:
            for row in item["spans"]:
                self.assertEqual(row["participant"], item["participant"])
                spans.append(coverage.Span(row["participant"], row["layer"], row["channel"],
                                           *row["spatial_Fortran_interval"]))
        self.assertEqual(coverage.verify_complete_coverage([*retained, *spans]), coverage.TOTAL)

    def test_exact_partial_progress_is_excluded_and_uncounted_future_rows_remain(self):
        prefix = tuple(coverage.retained_spans())
        partial = coverage.Span("sub-01", "conv1_1", 1, 0, 128)
        plan = plan_from_spans((*prefix, partial))
        self.assertEqual(plan["retained_units"], 51456)
        self.assertEqual(plan["remaining_units"], 74254984)
        first = plan["work_items"][0]["spans"][0]
        self.assertEqual(first["channel"], 1)
        self.assertEqual(first["spatial_Fortran_interval"], [128, 50176])
        completed = plan_from_spans((*prefix, *coverage.next_layer_spans()))
        self.assertEqual(completed["retained_units"], 3212416)
        self.assertEqual(completed["remaining_units"], 71094024)
        self.assertFalse(any(row["participant"] == "sub-01" and row["layer"] == "conv1_1"
                             for item in completed["work_items"] for row in item["spans"]))

    def test_duplicate_or_overlapping_retained_rows_never_become_a_plan(self):
        retained = tuple(coverage.retained_spans())
        with self.assertRaisesRegex(coverage.prefix.Stop, "overlap_or_duplicate"):
            remaining_after((*retained, retained[0]))
        with self.assertRaisesRegex(coverage.prefix.Stop, "overlap_or_duplicate"):
            remaining_after((*retained, coverage.Span("sub-01", "conv1_1", 0, 63, 65)))
        with self.assertRaises(coverage.prefix.Stop):
            plan_from_spans(retained, max_units=1_003_521)
        with self.assertRaises(coverage.prefix.Stop):
            plan_from_spans(retained, minutes=481)

    def test_many_channel_metadata_fits_in_addition_to_fragment_reservation(self):
        plan = plan_from_spans(tuple(coverage.retained_spans()))
        item = max(plan["work_items"], key=lambda value: len(value["spans"]))
        spans = item["spans"]
        self.assertGreater(len(spans), 1000)
        request = {"requested_spans": spans, "committed_spans": [], "remaining_spans": spans}
        status = {"requested_spans": spans, "committed_spans": spans, "remaining_spans": [],
                  "channel_timings": [{**span, "loading_seconds": 1000., "fitting_seconds": 1000.}
                                      for span in spans]}
        # Includes a further64KiB for fixed fields/errors/allocation metadata.
        metadata = sum(len(json.dumps(record, indent=2).encode()) + 1 for record in (request, status)) + 65_536
        fragments = item["units_requested"] * 6045 + len(spans) * 1_048_576
        self.assertLessEqual(fragments + metadata, item["resources"]["durable_bytes"])


if __name__ == "__main__":
    unittest.main()
