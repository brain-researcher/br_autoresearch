"""Source-free checks for the frozen observed-data report arithmetic."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch

import summarize_observed_behavior as summary


class ObservedBehaviorSummaryTests(unittest.TestCase):
    def setUp(self):
        self.implementation = {"fixture": "original implementation", "optimizer": {"maximum_sweeps": 500}}
        self.people = [f"sub-{index:03d}" for index in range(108)]

    def component(self, model, direction):
        scores = {}
        for index, person in enumerate(self.people):
            total = (-256 - 4*index) if direction == 0 else (-768 - 12*index)
            if model == "MZ":
                total += 2 + index % 3
                if direction == 1 and index >= 54:
                    total += 4
            scores[person] = {"version": "EI" if index < 54 else "ER", "heldout_trials": 128,
                              "log_sum": total, "contribution_sum_over_256": total/256}
        return {"model": model, "direction": direction, "status": "direction_completed",
                "source_validated": True, "direction_complete": True, "scores_available": True,
                "workers_cleaned_up": True, "source": str(summary.SOURCE),
                "implementation": deepcopy(self.implementation),
                "training_runs": list(summary.HALVES[direction][0]),
                "scoring_runs": list(summary.HALVES[direction][1]),
                "participants": list(self.people), "scored_participants": 108, "scored_trials": 13824,
                "participant_scores": scores,
                "fit": {"model": model, "rival": "fixed", "converged": True,
                        "subject_optimizer_failures": [], "participants": list(self.people),
                        "training_runs": list(summary.HALVES[direction][0]),
                        "scoring_runs": list(summary.HALVES[direction][1]),
                        "objective_history": [100., 100.],
                        "support_contacts": [{"fixture_contact": model}]}}

    def summarize(self, reports, duplicate=False):
        # Paths remain in the declared tree, but all report reads are intercepted.
        # No actual observed source, fit artifact or implementation is opened.
        records = {(summary.RUNS / "unexecuted-summary-fixture" / str(index) / "observed_behavior_fit.json").resolve(): report
                   for index, report in enumerate(reports)}
        paths = list(records)
        if duplicate:
            paths.append(paths[0])
        def read(path, *args, **kwargs):
            self.assertIn(path, records)
            return json.dumps(records[path])
        with patch.object(Path, "read_text", autospec=True, side_effect=read) as reader:
            result = summary.summarize(paths, self.implementation)
            self.assertEqual(reader.call_count, len(paths))
        return result

    def test_two_halves_sum_over_256_and_pair_people_before_summarizing(self):
        reports = [self.component(model, direction) for model in ("M0", "MZ") for direction in (0, 1)]
        result = self.summarize(reports)
        expected_base = [-4 - index/16 for index in range(108)]
        expected_delta = [(4 + 2*(index % 3) + (4 if index >= 54 else 0))/256 for index in range(108)]
        base = result["models"]["M0"]
        self.assertTrue(base["available"])
        self.assertEqual([base["participant_scores"][p]["nats_per_trial"] for p in self.people], expected_base)
        self.assertEqual(base["summary"]["mean_nats_per_trial"], statistics.fmean(expected_base))
        contrast = result["paired_differences_from_M0"]["MZ-M0"]
        self.assertTrue(contrast["available"])
        self.assertEqual(list(contrast["participant_differences"].values()), expected_delta)
        self.assertEqual(contrast["summary"]["participant_SE"], statistics.stdev(expected_delta)/math.sqrt(108))
        self.assertEqual(contrast["summary"]["participant_median"], statistics.median(expected_delta))
        self.assertEqual(contrast["summary"]["version_means"],
                         {"EI": statistics.fmean(expected_delta[:54]), "ER": statistics.fmean(expected_delta[54:])})
        self.assertEqual(base["directions"][0]["fit_support_contacts"], reports[0]["fit"]["support_contacts"])
        self.assertEqual((base["directions"][0]["converged"], base["directions"][0]["sweeps"]), (True, 2))
        self.assertEqual(result["logistic_benchmark"], "registered_but_not_implemented_in_this_report")
        self.assertNotIn("winner", result)

    def test_missing_or_technical_half_never_promotes_a_model_or_pair(self):
        empty = self.summarize([])
        self.assertEqual(set(empty["models"]), set(summary.MODELS))
        self.assertTrue(all(not row["available"] for row in empty["models"].values()))
        reports = [self.component("M0", 0), self.component("M0", 1), self.component("MZ", 0)]
        result = self.summarize(reports)
        self.assertTrue(result["models"]["M0"]["available"])
        self.assertFalse(result["models"]["MZ"]["available"])
        self.assertFalse(result["paired_differences_from_M0"]["MZ-M0"]["available"])
        technical = {"model": "MZ", "direction": 1, "status": "technical_failure", "error": "time cap"}
        result = self.summarize(reports + [technical])
        self.assertFalse(result["models"]["MZ"]["available"])
        self.assertNotIn("participant_scores", result["models"]["MZ"])
        self.assertEqual(result["models"]["MZ"]["directions"][1]["reason"], "time cap")
        no_base = self.summarize([self.component("MZ", 0), self.component("MZ", 1)])
        self.assertTrue(no_base["models"]["MZ"]["available"])
        self.assertTrue(all(not row["available"] for row in no_base["paired_differences_from_M0"].values()))

    def test_duplicate_identity_split_spec_and_denominator_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "Duplicate attempt"):
            self.summarize([self.component("M0", 0)], duplicate=True)
        def versions(report):
            report["participant_scores"][self.people[0]]["version"] = "ER"
            report["participant_scores"][self.people[54]]["version"] = "EI"
        mutations = [versions,
                     lambda r: r.update(source="not-the-frozen-source"),
                     lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
                     lambda r: r.update(training_runs=[1, 2]),
                     lambda r: r["participant_scores"][self.people[0]].update(heldout_trials=127),
                     lambda r: r["participant_scores"][self.people[0]].update(contribution_sum_over_256=-999),
                     lambda r: r["fit"].update(converged=False),
                     lambda r: r["fit"].update(subject_optimizer_failures=[{"participant": self.people[0]}]),
                     lambda r: r["fit"].update(objective_history=[100.]*501),
                     lambda r: r["participants"].__setitem__(0, "sub-other")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                reports = [self.component("M0", 0), self.component("M0", 1)]
                mutation(reports[1])
                with self.assertRaises(ValueError):
                    self.summarize(reports)


if __name__ == "__main__":
    unittest.main()
