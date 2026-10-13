"""Known-answer point-bias checks; authored reports only, no model fitting."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import summarize_mz_group_recovery as core


PEOPLE = [f"fixture-{i:03}" for i in range(108)]


def report_fixture(spec, index=0, *, core_pass=True):
    flags = {"mean_gain_at_least_0p01": core_pass,
             "mean_gain_greater_than_two_participant_SE": True,
             "positive_in_both_task_versions": True,
             "positive_participant_median": True,
             "participant_median_95CI_lower_positive": True}
    fits = []
    for training, scoring in (((1, 4), (2, 3)), ((2, 3), (1, 4))):
        for model in ("M0", "MZ"):
            names = ["kappa", "log_a", "nondecision_logit"] + (["rho"] if model == "MZ" else [])
            center = [spec["parameters"][name]["population_center"] for name in names]
            if model == "MZ":
                center[-1] = math.log(.45/.55)
            fits.append({"model": model, "rival": "fixed", "training_runs": list(training),
                "scoring_runs": list(scoring), "participants": PEOPLE[:], "parameter_names": names,
                "parameters": [center[:] for _ in PEOPLE], "global_mean": center[:],
                "version_means": [center[:], center[:]],
                "version_standard_deviations": [[.3] * len(names), [.3] * len(names)],
                "version_order": ["EI", "ER"], "inference": spec["method"],
                "converged": True, "subject_optimizer_failures": [], "objective_history": [100., 100.],
                "support_contacts": [], "group_interval_coverage_qualified": False,
                "individual_rank_recovery_qualified": False})
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_fitted_recovery_component",
        "generating_model": "MZ", "replicate_index": index, "planned_cell_replicates": 200,
        "seed": 910031 + 10007 * index, "SESOI_level": 1., "implementation": spec,
        "focal_generating_truth": {"eta": 0., "rho": math.log(.45/.55), "criterion": 0.},
        "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
        "status": "necessary_primary_comparison_completed", "valid_recovery_component": True,
        "counts_as_curtailment_failure": not core_pass, "finished_utc": "2026-10-05T00:00:00+00:00",
        "participants": 108, "runs": 432, "trials": 27648,
        "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        "focal_effect_between_participant_variance": 0., "fits": fits,
        "participant_scores": {p: {"version": "EI" if i < 54 else "ER"} for i, p in enumerate(PEOPLE)},
        "comparison": {"focal_model": "MZ", "requirements": flags,
            "core_family_selection_pass": core_pass, "counts_as_curtailment_failure": not core_pass,
            "necessary_primary_comparison_pass": core_pass, "richer_primary_diagnostic_pass": core_pass,
            "full_recovery_pass": None}}


def set_z(report, *, ei=.45, er=.45):
    for fit in report["fits"]:
        if fit["model"] == "MZ":
            for i, row in enumerate(fit["parameters"]):
                z = ei if i < 54 else er
                row[-1] = math.log(z / (1 - z))


def indexed(template, index):
    # Immutable shared fit payload avoids manufacturing200 redundant fixtures.
    return dict(template, replicate_index=index, seed=910031 + 10007 * index)


class MZGroupRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())

    def summarize(self, *reports):
        return core.summarize_group_reports([(f"fixture-{i}.json", report) for i, report in enumerate(reports)], self.spec)

    def test_physical_z_is_averaged_over_people_and_both_halves_not_logit_means(self):
        report = report_fixture(self.spec)
        for fit in report["fits"]:
            if fit["model"] == "MZ":
                first = fit["training_runs"] == [1, 4]
                for i, row in enumerate(fit["parameters"]):
                    z = (.2 if i % 2 == 0 else .6) if i < 54 and first else (.5 if i < 54 else .4)
                    row[-1] = math.log(z/(1-z))
                # Hypermeans deliberately disagree with the individual means.
                fit["global_mean"][-1] = 1.
                for row in fit["version_means"]:
                    row[-1] = 1.
        result = self.summarize(report)
        self.assertAlmostEqual(result["version_statistics"]["EI"]["mean_rejection_displacement"], .05)
        self.assertAlmostEqual(result["version_statistics"]["ER"]["mean_rejection_displacement"], .10)
        self.assertTrue(result["statistics_provisional"])
        self.assertIsNone(result["both_versions_relative_bias_pass"])

    def test_valid_core_failure_is_kept_and_support_contacts_do_not_exclude(self):
        report = report_fixture(self.spec, core_pass=False)
        fit = report["fits"][1]
        low, high = self.spec["parameters"]["rho"]["bounds"]
        fit["parameters"][0][-1] = low
        fit["support_contacts"] = [{"level": "participant", "participant": PEOPLE[0],
                                    "parameter": "rho", "value": low, "bounds": [low, high]}]
        result = self.summarize(report)
        self.assertEqual(result["valid_replicates"], 1)
        self.assertEqual(result["core_selection_failures_included"], 1)
        self.assertEqual(result["valid_reports"][0]["fit_support_contact_counts"][1]["count"], 1)
        expected = .5 - (107 * .45 + 1/(1+math.exp(-low))) / 108
        self.assertAlmostEqual(result["version_statistics"]["EI"]["mean_rejection_displacement"], expected)

    def test_no_final_verdict_at199_even_when_all_point_estimates_are_known(self):
        template = report_fixture(self.spec)
        result = self.summarize(*(indexed(template, i) for i in range(199)))
        self.assertEqual(result["valid_replicates"], 199)
        self.assertEqual(result["remaining_unique_valid_replicates"], 1)
        self.assertFalse(result["group_point_bias_complete"])
        self.assertIsNone(result["version_statistics"]["EI"]["relative_bias_pass"])

    def test_complete200_bias_is_bias_of_mean_not_mean_absolute_error(self):
        low, high = report_fixture(self.spec), report_fixture(self.spec, core_pass=False)
        set_z(low, ei=.43, er=.43)
        set_z(high, ei=.47, er=.47)
        result = self.summarize(*(indexed(low if i % 2 else high, i) for i in range(200)))
        self.assertTrue(result["group_point_bias_complete"])
        self.assertTrue(result["both_versions_relative_bias_pass"])
        self.assertAlmostEqual(result["version_statistics"]["EI"]["absolute_relative_bias"], 0.)
        self.assertEqual(result["core_selection_failures_included"], 100)
        self.assertFalse(result["interval_coverage_evaluated"])
        self.assertFalse(result["group_interval_coverage_qualified"])
        self.assertEqual(result["full_recovery_gate"], "not_evaluated")

    def test_denominator_is_effect_005_and_both_versions_must_pass(self):
        template = report_fixture(self.spec)
        set_z(template, ei=.44, er=.45)
        result = self.summarize(*(indexed(template, i) for i in range(200)))
        self.assertAlmostEqual(result["version_statistics"]["EI"]["absolute_relative_bias"], .2)
        self.assertFalse(result["version_statistics"]["EI"]["relative_bias_pass"])
        self.assertTrue(result["version_statistics"]["ER"]["relative_bias_pass"])
        self.assertFalse(result["both_versions_relative_bias_pass"])

    def test_technical_attempts_remain_unresolved_and_never_supply_partial_fits(self):
        technical = report_fixture(self.spec)
        technical.update(status="technical_recovery_failure", valid_recovery_component=False,
                         counts_as_curtailment_failure=False, fits=[{"parameters": "closed partial state"}])
        result = self.summarize(technical)
        self.assertEqual(result["valid_replicates"], 0)
        self.assertEqual(result["remaining_unique_valid_replicates"], 200)
        self.assertEqual(len(result["technical_attempts"]), 1)
        self.assertIsNone(result["version_statistics"]["ER"]["mean_rejection_displacement"])
        self.assertIsNone(result["both_versions_relative_bias_pass"])

    def test_duplicate_index_and_unfinished_checkpoint_are_rejected(self):
        original = report_fixture(self.spec)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.summarize(original, deepcopy(original))
        for change in ({"finished_utc": None}, {"finished_utc": "2026-10-05T00:00:00"}, {"status": "running"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.summarize(dict(original, **change))

    def test_wrong_truth_seed_implementation_and_false_interval_claim_are_rejected(self):
        for kind in ("truth", "seed", "implementation", "interval"):
            report = report_fixture(self.spec)
            if kind == "truth":
                report["focal_generating_truth"]["rho"] = 0.
            elif kind == "seed":
                report["seed"] += 1
            elif kind == "implementation":
                report["implementation"] = dict(self.spec, policy_version=99)
            else:
                report["fits"][1]["group_interval_coverage_qualified"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.summarize(report)

    def test_nonfinite_coerced_outside_support_and_unconverged_fits_are_rejected(self):
        for value in (float("nan"), float("inf"), "0.1", True, 4.):
            report = report_fixture(self.spec)
            report["fits"][1]["parameters"][0][-1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.summarize(report)
        report = report_fixture(self.spec)
        report["fits"][3]["converged"] = False
        with self.assertRaises(RuntimeError):
            self.summarize(report)
        report = report_fixture(self.spec)
        report["fits"][0]["version_means"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.summarize(report)

    def test_missing_duplicate_half_changed_roster_and_unbalanced_versions_rejected(self):
        for kind in ("missing", "duplicate", "reordered", "unbalanced", "cohort_size"):
            report = report_fixture(self.spec)
            if kind == "missing":
                report["fits"].pop()
            elif kind == "duplicate":
                report["fits"][3] = deepcopy(report["fits"][1])
            elif kind == "reordered":
                report["fits"][1]["participants"].reverse()
            elif kind == "unbalanced":
                report["participant_scores"][PEOPLE[0]]["version"] = "ER"
            else:
                report["participants"] = 107
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.summarize(report)
        original, changed = report_fixture(self.spec), report_fixture(self.spec, index=1)
        changed["participant_scores"][PEOPLE[0]]["version"] = "ER"
        changed["participant_scores"][PEOPLE[54]]["version"] = "EI"
        with self.assertRaisesRegex(ValueError, "across replicates"):
            self.summarize(original, changed)


if __name__ == "__main__":
    unittest.main()
