"""Small arithmetic fixtures; no empirical records or scientific simulations."""
import copy
import itertools
import math
import statistics
import unittest

import analyze_observed_model_contrasts as contrasts


def fixture():
    source = {"analysis_id": "exploratory_observed_behavior", "denominator": contrasts.DENOMINATOR,
              "logistic_benchmark": "registered_but_not_implemented_in_this_report", "models": {}}
    for model_index, model in enumerate(contrasts.MODELS):
        # Reverse insertion order in alternate models to detect positional pairing.
        indices = range(108) if model_index % 2 else reversed(range(108))
        scores = {f"sub-{i:03d}": {"version": "EI" if i < 54 else "ER",
                                    "nats_per_trial": -20 - i/4 + model_index * (i - 70)/128}
                  for i in indices}
        source["models"][model] = {"available": True, "participant_scores": scores,
                                    "directions": [{"complete": True, "fit_support_contacts": ["retained"]},
                                                   {"complete": True, "fit_support_contacts": []}]}
    return source


class ObservedContrastTests(unittest.TestCase):
    def test_complete_menu_pairs_identities_and_keeps_negative_values(self):
        source = fixture()
        untouched = copy.deepcopy(source)
        result = contrasts.analyze(source)
        self.assertEqual(source, untouched)
        self.assertEqual(set(result["models"]), set(contrasts.MODELS))
        self.assertEqual(len(result["contrasts"]), 15)
        self.assertEqual({frozenset(pair) for pair in contrasts.PAIRS},
                         {frozenset(pair) for pair in itertools.combinations(contrasts.MODELS, 2)})
        row = result["contrasts"]["MV-MZ"]
        expected = [(70 - i)/128 for i in range(108)]
        self.assertEqual([r["nats_per_trial"] for r in row["participant_differences"].values()], expected)
        summary = row["summary"]
        self.assertEqual(summary["mean_nats_per_trial"], statistics.fmean(expected))
        self.assertEqual(summary["participant_SE"], statistics.stdev(expected)/math.sqrt(108))
        self.assertEqual(summary["positive_participants"], 70)
        self.assertEqual(summary["zero_participants"], 1)
        self.assertEqual(summary["negative_participants"], 37)
        self.assertEqual(row["by_version"]["EI"]["mean_nats_per_trial"], statistics.fmean(expected[:54]))
        self.assertEqual(row["by_version"]["ER"]["mean_nats_per_trial"], statistics.fmean(expected[54:]))
        self.assertEqual(result["models"]["MV"]["directions"], source["models"]["MV"]["directions"])

    def test_leave_one_participant_ranges_are_paired_summary_not_refits(self):
        row = contrasts.analyze(fixture())["contrasts"]["MVC-MV"]["summary"]
        expected = [4 * (i - 70)/128 for i in range(108)]
        omitted = [statistics.fmean(expected[:i] + expected[i+1:]) for i in range(108)]
        self.assertEqual(row["leave_one_participant_out_mean_range"], [min(omitted), max(omitted)])
        self.assertEqual(row["mean_nats_per_trial"], statistics.fmean(expected))

    def test_incomplete_or_misaligned_support_is_never_silently_filtered(self):
        mutations = [
            lambda s: s["models"]["MV"].update(available=False),
            lambda s: s["models"]["MV"]["directions"].pop(),
            lambda s: s["models"]["MV"]["participant_scores"].pop("sub-001"),
            lambda s: s["models"]["MV"]["participant_scores"]["sub-001"].update(nats_per_trial=float("nan")),
            lambda s: s.update(denominator="128 trials"),
        ]
        for mutate in mutations:
            source = fixture()
            mutate(source)
            with self.assertRaises(ValueError):
                contrasts.analyze(source)
        source = fixture()
        source["models"]["MV"]["participant_scores"]["sub-001"]["version"] = "ER"
        source["models"]["MV"]["participant_scores"]["sub-055"]["version"] = "EI"
        with self.assertRaisesRegex(ValueError, "identities differ"):
            contrasts.analyze(source)


if __name__ == "__main__":
    unittest.main()
