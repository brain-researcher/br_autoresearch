"""Source-free checks for retained halves and exact chronology constraints."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_source_chronology as clock_stage
import qualify_source_roles as stage
import reconcile_released_events as core


def row(subject="sub-02", session=1, run=1, order=0, image="a", repetition=1, gap="", index=None):
    return {"participant": subject, "session": str(session), "run": str(run),
            "source_order": str(order), "source_repetition": str(repetition),
            "trials_since_lastrep": str(gap), "glmsingle_trial_index_zero_based": str(run * 60 + order if index is None else index),
            "expected_provider_image_path": f"images_fmri/{image}/{image}.jpg"}


def four_person_pair(ordinals=(1, 2), gap=1):
    rows = [row(subject=subject, order=order, repetition=repetition, gap="" if order == 0 else gap)
            for subject in stage.SUBJECTS for order, repetition in enumerate(ordinals)]
    return rows, {rows[0]["expected_provider_image_path"]: True}


def edge(a, b, delta, previous_index=0, current_index=1):
    return (a, b, delta, previous_index, current_index)


def connected_four_person_rows():
    rows, eligible = [], {}
    for subject in stage.SUBJECTS:
        sequence = clock_stage.KNOWN_SUB01_SES17 if subject == "sub-01" else (1, 2)
        session = 17 if subject == "sub-01" else 1
        for i, (a, b) in enumerate(zip(sequence, sequence[1:])):
            image = f"bridge_{i}"
            # Consecutive runs have starts separated by60; order20→10 gives gap50.
            rows.extend([row(subject, session, a, 20, image, 1), row(subject, session, b, 10, image, 2, 50)])
            eligible[rows[-1]["expected_provider_image_path"]] = False
    return rows, eligible


class RepeatMappingTests(unittest.TestCase):
    def test_gap_ordinals_use_retained_parity_and_no_unjustified_edge(self):
        rows, eligible = four_person_pair((1, 3))
        result = clock_stage.repeat_metadata(rows, eligible, clock_stage.Issues())
        mapped, edges = result[:2]
        for subject in stage.SUBJECTS:
            selected = [r for r in rows if r["participant"] == subject]
            self.assertEqual([mapped[stage.key(r)]["repeat_half"] for r in selected], [1, 2])
            self.assertEqual(edges[subject], [])
        self.assertTrue(result[5])
        self.assertEqual(result[6]["skipped_nonconsecutive_source_ordinal_pairs"], 4)

    def test_absent_ineligible_image_is_not_bad_ordinal_and_gets_no_half(self):
        rows, eligible = four_person_pair()
        extra = row("sub-01", image="ineligible", order=5)
        rows.append(extra)
        eligible[extra["expected_provider_image_path"]] = False
        issues = clock_stage.Issues()
        result = clock_stage.repeat_metadata(rows, eligible, issues)
        self.assertTrue(result[5])
        self.assertEqual(result[4], set())
        self.assertEqual(result[0][stage.key(extra)]["repeat_half"], "")
        self.assertEqual(result[0][stage.key(extra)]["retained_repeat_rank"], 0)
        self.assertEqual(dict(issues.counts), {})

    def test_duplicate_or_missing_ordinal_invalidates_only_observed_group(self):
        for bad in ("1", ""):
            rows, eligible = four_person_pair()
            rows[3]["source_repetition"] = bad
            result = clock_stage.repeat_metadata(rows, eligible, clock_stage.Issues())
            self.assertFalse(result[5])
            self.assertEqual(result[0][stage.key(rows[2])]["repeat_half"], "")
            self.assertEqual(result[0][stage.key(rows[0])]["repeat_half"], 1)

    def test_within_run_ordinal_inversion_is_not_sorted_away(self):
        rows, eligible = four_person_pair((2, 1))
        issues = clock_stage.Issues()
        result = clock_stage.repeat_metadata(rows, eligible, issues)
        self.assertFalse(result[5])
        self.assertEqual(issues.counts["ordinal_within_run_order_conflict"], 4)

    def test_exact_trial_gap_equation_and_missing_predecessor_scope(self):
        rows = [row(run=1, order=7, repetition=1), row(run=2, order=9, repetition=2, gap=122),
                row(run=3, order=2, repetition=4, gap=5)]
        eligible = {rows[0]["expected_provider_image_path"]: False}
        result = clock_stage.repeat_metadata(rows, eligible, clock_stage.Issues())
        self.assertEqual(len(result[1]["sub-02"]), 1)
        self.assertEqual(result[1]["sub-02"][0][2], 120)
        self.assertEqual(result[6]["skipped_nonconsecutive_source_ordinal_pairs"], 1)

    def test_unknown_lag_does_not_fabricate_an_edge_or_invalidate_halves(self):
        rows, eligible = four_person_pair(gap="")
        result = clock_stage.repeat_metadata(rows, eligible, clock_stage.Issues())
        self.assertTrue(result[5])
        self.assertEqual(sum(map(len, result[1].values())), 0)
        self.assertEqual(result[6]["skipped_unknown_trial_gap_pairs"], 4)


class GraphTests(unittest.TestCase):
    def test_integer_cycle_and_gap_preservation_without_numeric_run_sort(self):
        a, b, c = [("sub-02", 1, n) for n in (1, 2, 3)]
        # Run2 actually precedes run1; run3 starts after a120-trial interval.
        runs, summary = clock_stage.solve_runs("sub-02", {a, b, c},
            [edge(b, a, 60), edge(a, c, 120), edge(b, c, 180)], clock_stage.Issues())
        self.assertTrue(summary["global_order_qualified"])
        self.assertEqual([runs[n]["global_source_history_start"] for n in (b, a, c)], [0, 60, 180])

    def test_cycle_conflict_and_nonzero_same_run_constraint_clear_positions(self):
        a, b, c = [("sub-02", 1, n) for n in (1, 2, 3)]
        for constraints in ([edge(a, b, 60), edge(b, c, 60), edge(a, c, 121)],
                            [edge(a, b, 60), edge(b, c, 60), edge(a, a, 1)]):
            runs, summary = clock_stage.solve_runs("sub-02", {a, b, c}, constraints, clock_stage.Issues())
            self.assertFalse(summary["global_order_qualified"])
            self.assertTrue(all(r["component_source_history_start"] == "" for r in runs.values()))

    def test_disconnection_does_not_get_global_tie_break(self):
        a, b, c = [("sub-02", 1, n) for n in (1, 2, 3)]
        runs, summary = clock_stage.solve_runs("sub-02", {a, b, c}, [edge(a, b, 60)], clock_stage.Issues())
        self.assertEqual(summary["connected_components"], 2)
        self.assertFalse(summary["global_order_qualified"])
        self.assertTrue(all(r["global_source_history_start"] == "" for r in runs.values()))
        self.assertTrue(all(r["component_coordinates_qualified"] for r in runs.values()))

    def test_run_intervals_must_not_overlap(self):
        a, b = [("sub-02", 1, n) for n in (1, 2)]
        runs, summary = clock_stage.solve_runs("sub-02", {a, b}, [edge(a, b, 59)], clock_stage.Issues())
        self.assertFalse(summary["global_order_qualified"])
        self.assertTrue(all(not r["component_coordinates_qualified"] for r in runs.values()))

    def test_documented_exception_accepts_actual_order_rejects_numeric_order(self):
        nodes = {("sub-01", 17, n) for n in range(1, 7)}
        for sequence, expected in ((clock_stage.KNOWN_SUB01_SES17, True), ((1, 2, 3, 4, 5, 6), False)):
            constraints = [edge(("sub-01", 17, a), ("sub-01", 17, b), 60) for a, b in zip(sequence, sequence[1:])]
            runs, summary = clock_stage.solve_runs("sub-01", nodes, constraints, clock_stage.Issues())
            self.assertEqual(summary["global_order_qualified"], expected)
            if not expected:
                self.assertTrue(all(r["component_source_history_start"] == "" for r in runs.values()))

    def test_component_ordinal_contradiction_detected_even_when_disconnected(self):
        rows, eligible = connected_four_person_rows()
        # An ordinal gap creates no edge, but still contradicts relative order
        # already fixed by another image in the connected component.
        rows.extend([row("sub-02", run=1, order=30, image="contradiction", repetition=3),
                     row("sub-02", run=2, order=30, image="contradiction", repetition=1),
                     row("sub-02", run=3, order=5, image="isolated", repetition=1)])
        eligible[rows[-2]["expected_provider_image_path"]] = False
        eligible[rows[-1]["expected_provider_image_path"]] = False
        projected, runs, report = clock_stage.chronology(rows, eligible)
        self.assertFalse(report["global_chronology_complete"])
        self.assertFalse(report["all_observed_ordinals_qualified"])
        self.assertEqual(report["issue_counts"]["ordinal_component_order_conflict"], 1)
        bad = [r for r in projected if "contradiction" in r["expected_provider_image_path"]]
        self.assertTrue(all(r["retained_repeat_rank"] == "" and r["repeat_half"] == "" for r in bad))
        self.assertTrue(all(r["component_source_history_position"] == "" for r in bad))

    def test_all_observed_rows_preserved_and_ineligible_halves_remain_empty(self):
        rows, eligible = connected_four_person_rows()
        projected, runs, report = clock_stage.chronology(rows, eligible)
        self.assertTrue(report["global_chronology_complete"])
        self.assertEqual(len(projected), len(rows))
        self.assertEqual({stage.key(r) for r in projected}, {stage.key(r) for r in rows})
        self.assertTrue(all(r["repeat_half"] == "" for r in projected))
        self.assertTrue(all(r["global_source_history_position"] != "" for r in projected))


class PublicationTests(unittest.TestCase):
    def test_deadline_after_computation_clears_every_completion_flag(self):
        rows, eligible = connected_four_person_rows()
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage, "load_sources", return_value=rows), \
                patch.object(clock_stage, "load_eligibility", return_value=eligible), \
                patch.object(stage, "write_table", side_effect=core.Refusal("attempt_deadline")):
            out = Path(temporary)
            self.assertEqual(clock_stage.run_attempt(out), 2)
            report = json.loads((out / "chronology_report.json").read_text())
            for flag in ("source_chronology_complete", "retained_half_qualification_complete", "global_chronology_complete"):
                self.assertFalse(report[flag])
            self.assertFalse(report["roles_assigned"])

    def test_issue_examples_are_bounded_while_counts_remain_complete(self):
        issues = clock_stage.Issues()
        for n in range(100):
            issues.add("fixture_problem", trial_index=n)
        self.assertEqual(issues.counts["fixture_problem"], 100)
        self.assertEqual(len(issues.examples), 50)


if __name__ == "__main__":
    unittest.main()
