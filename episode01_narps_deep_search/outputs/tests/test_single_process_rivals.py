"""Meaningful saved-cohort and paired-scoring checks with no fitted experiment."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_single_process_rivals as core
from hierarchical_behavior import parameter_names
from test_single_process_recovery import cohort


THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


class SingleProcessRivalsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())
        cls.people = {person: [dict(row, choice="1", no_response="0", strength="-1",
                                  response_time="1.0", response_category="simulated_accept") for row in rows]
                      for person, rows in cohort().items()}

    def fit(self, model, train):
        names = parameter_names(model, "fixed")
        theta = [self.spec["parameters"][name]["population_center"] for name in names]
        theta[0] = {"M0": (.2, .4), "MZ": (.6, .8), "MC": (.5, .7), "MV": (.9, 1.1)}[model][0 if train == (1, 4) else 1]
        return {"model": model, "rival": "fixed", "participants": sorted(self.people), "parameter_names": names,
            "parameters": [theta[:] for _ in self.people], "inference": self.spec["method"], "version_order": ["EI", "ER"],
            "global_mean": theta[:], "version_means": [theta[:], theta[:]],
            "version_standard_deviations": [[.3] * len(names), [.3] * len(names)],
            "converged": True, "subject_optimizer_failures": [], "support_contacts": [],
            "objective_history": [10000., 10000.00001], "training_runs": list(train),
            "scoring_runs": [2, 3] if train == (1, 4) else [1, 4]}

    @staticmethod
    def likelihood(theta, names, data, spec, rival):
        return np.full(len(data["choice"]), theta[0])

    def fixture(self, directory):
        directory.mkdir()
        events = directory / "simulated_events.csv.gz"
        with gzip.open(events, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=core.DESIGN_FIELDS + core.RESPONSE_FIELDS)
            writer.writeheader()
            writer.writerows(row for rows in self.people.values() for row in rows)
        truth = {person: {**{name: self.spec["parameters"][name]["population_center"]
                            for name in ("kappa", "log_a", "nondecision_logit")}, **core.focal_truth("MZ")}
                 for person in self.people}
        requirements = {name: True for name in ("mean_gain_at_least_0p01", "mean_gain_greater_than_two_participant_SE",
            "positive_in_both_task_versions", "positive_participant_median", "participant_median_95CI_lower_positive")}
        # The same two ordered sum/256 operations as the saved runner, with no fit.
        scores = {person: {"M0": float(np.full(128, .2).sum()) / 256 + float(np.full(128, .4).sum()) / 256,
                           "MZ": float(np.full(128, .6).sum()) / 256 + float(np.full(128, .8).sum()) / 256,
                           "version": rows[0]["task_version"]} for person, rows in self.people.items()}
        report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_fitted_recovery_component",
            "generating_model": "MZ", "replicate_index": 0, "seed": 910031, "SESOI_level": 1., "planned_cell_replicates": 200,
            "focal_generating_truth": core.focal_truth("MZ"), "implementation": self.spec,
            "status": "necessary_primary_comparison_completed", "valid_recovery_component": True,
            "counts_as_curtailment_failure": False, "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
            "started_utc": "2026-10-05T00:00:00+00:00", "finished_utc": "2026-10-05T00:01:00+00:00",
            "participants": 108, "runs": 432, "trials": 27648, "simulated_deadline_misses": 0,
            "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
            "design_source": str(core.DESIGN_SOURCE), "design_fields": list(core.DESIGN_FIELDS),
            "fits": [self.fit(model, train) for train, _ in core.HALVES for model in ("M0", "MZ")],
            "participant_scores": scores, "comparison": {"focal_model": "MZ", "requirements": requirements,
                "core_family_selection_pass": True, "counts_as_curtailment_failure": False,
                "necessary_primary_comparison_pass": True, "richer_primary_diagnostic_pass": True, "full_recovery_pass": None}}
        (directory / "single_process_recovery.json").write_text(json.dumps(report))
        (directory / "generating_parameters.json").write_text(json.dumps(truth))
        return report, truth, events

    def execute(self, directory, fitter=None, likelihood=None):
        source = directory / "source"
        self.fixture(source)
        args = argparse.Namespace(source_dir=source, implementation=core.IMPLEMENTATION,
                                  output_dir=directory / "rivals", max_seconds=60)
        def default_fitter(training, model, rival, spec):
            self.assertIn(model, ("MC", "MV"))
            self.assertEqual(rival, "fixed")
            self.assertEqual(spec, self.spec)
            self.assertEqual(set(training), set(self.people))
            runs = tuple(sorted({int(row["run"]) for row in training[next(iter(training))]["rows"]}))
            self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
            self.assertIn(runs, ((1, 4), (2, 3)))
            return self.fit(model, runs)
        with patch.object(core, "SOURCE", source), patch.object(core, "require_compute"), patch.dict(core.os.environ, THREADS), \
                patch.object(core, "fit_partially_pooled", side_effect=fitter or default_fitter) as fits, \
                patch.object(core, "trial_likelihood", side_effect=likelihood or self.likelihood):
            result = core.run(args)
            with self.assertRaises(FileExistsError):
                core.run(args)
        return result, fits, args

    def test_saved_synthetic_header_and_response_validation_without_sequence_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, path = self.fixture(Path(temporary) / "source")
            self.assertEqual(core.load_cohort(path), self.people)
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=core.DESIGN_FIELDS + core.RESPONSE_FIELDS)
                writer.writeheader()
                row = dict(self.people["synthetic-000"][0], choice="-1", no_response="1", response_category="no_response")
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "deadline miss"):
                core.load_cohort(path)

    def test_saved_provenance_truth_and_complete_fit_support_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, truth, _ = self.fixture(Path(temporary) / "source")
            core.validate_source(report, truth, self.people, self.spec)
            for field, value in (("seed", 910032), ("replicate_index", 1), ("finished_utc", None),
                                 ("valid_recovery_component", False), ("observed_choices_or_RTs_used", True)):
                changed = copy.deepcopy(report)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    core.validate_source(changed, truth, self.people, self.spec)
            changed = copy.deepcopy(report)
            changed["fits"][1]["converged"] = False
            with self.assertRaisesRegex(RuntimeError, "Nonconverged"):
                core.validate_source(changed, truth, self.people, self.spec)
            changed_truth = copy.deepcopy(truth)
            changed_truth["synthetic-000"]["rho"] = 0.
            with self.assertRaisesRegex(ValueError, "selected MZ truth"):
                core.validate_source(report, changed_truth, self.people, self.spec)

    def test_saved_score_replay_preserves_baseline_accepts_roundoff_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, _, events = self.fixture(Path(temporary) / "source")
            splits = [core.load_split(events, list(train), list(held)) for train, held in core.HALVES]
            with patch.object(core, "trial_likelihood", side_effect=self.likelihood), patch.object(core, "fit_partially_pooled") as fitter:
                saved, discrepancy = core.replay_saved_scores(report, splits, self.spec)
                self.assertEqual(saved, report["participant_scores"])
                self.assertEqual(discrepancy, 0.)
                report["participant_scores"]["synthetic-000"]["MZ"] += 1e-14
                saved, discrepancy = core.replay_saved_scores(report, splits, self.spec)
                self.assertEqual(saved, report["participant_scores"])
                self.assertGreater(discrepancy, 0.)
                self.assertLess(discrepancy, core.REPLAY_ABSOLUTE_TOLERANCE)
                report["participant_scores"]["synthetic-000"]["MZ"] += 1e-8
                with self.assertRaisesRegex(ValueError, "replay tolerance"):
                    core.replay_saved_scores(report, splits, self.spec)
                fitter.assert_not_called()

    def test_paired_contrast_sign_and_participant_SE_use_the_same_people(self):
        effects = np.linspace(-.2, .4, 108)
        scores = {person: {"M0": -1., "MZ": -.9, "MC": -.9 - effect, "version": rows[0]["task_version"]}
                  for (person, rows), effect in zip(self.people.items(), effects)}
        result = core.rival_diagnostics(scores, "MC")
        contrast = result["saved_MZ_minus_rival"]
        self.assertAlmostEqual(contrast["mean_gain_nats_per_trial"], float(effects.mean()))
        self.assertAlmostEqual(contrast["participant_SE"], float(effects.std(ddof=1) / np.sqrt(108)))
        self.assertAlmostEqual(contrast["version_means"]["EI"], float(effects[:54].mean()))
        self.assertAlmostEqual(contrast["participant_median"], float(np.median(effects)))
        self.assertEqual(result["rival_vs_saved_M0"]["fitted_model"], "MC")
        self.assertIsNone(result["rival_vs_saved_M0"]["full_recovery_pass"])
        self.assertNotIn("counts_as_curtailment_failure", result["rival_vs_saved_M0"])

    def test_exact_four_fit_order_halves_score_denominator_and_no_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, args = self.execute(Path(temporary))
            self.assertEqual([call.args[1] for call in calls.call_args_list], ["MC", "MC", "MV", "MV"])
            self.assertEqual([fit["training_runs"] for fit in result["fits"]], [[1, 4], [2, 3], [1, 4], [2, 3]])
            self.assertTrue(result["rival_diagnostics_complete"])
            self.assertTrue(result["saved_scores_replayed"])
            self.assertFalse(result["simulation_regenerated"])
            self.assertFalse(result["saved_M0_MZ_refitted"])
            for scores in result["participant_scores"].values():
                self.assertAlmostEqual(scores["M0"], .3)
                self.assertAlmostEqual(scores["MZ"], .7)
                self.assertAlmostEqual(scores["MC"], .6)
                self.assertAlmostEqual(scores["MV"], 1.)
            for key in ("family_selection", "confusion_rate", "group_recovery", "full_gate"):
                self.assertEqual(result[key], "not_evaluated")
            self.assertEqual(json.loads((args.output_dir / core.REPORT_NAME).read_text()), result)
            self.assertFalse((args.output_dir / "simulated_events.csv.gz").exists())

    def test_later_timeout_keeps_completed_MC_and_checkpoints_without_claiming_completion(self):
        def fitter(training, model, rival, spec):
            if model == "MV":
                core.interrupted(signal.SIGTERM, None)
            runs = tuple(sorted({int(row["run"]) for row in training[next(iter(training))]["rows"]}))
            return self.fit(model, runs)
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, args = self.execute(Path(temporary), fitter=fitter)
            self.assertEqual(result["status"], "technical_rival_failure")
            self.assertFalse(result["rival_diagnostics_complete"])
            self.assertEqual(result["completed_rivals"], ["MC"])
            self.assertEqual(len(result["fits"]), 2)
            self.assertEqual(set(result["diagnostics"]), {"MC"})
            self.assertEqual(calls.call_count, 3)
            self.assertNotIn("MV", next(iter(result["participant_scores"].values())))
            self.assertTrue(result["finished_utc"])
            self.assertEqual(json.loads((args.output_dir / core.REPORT_NAME).read_text()), result)

    def test_nonconverged_fit_is_preserved_but_never_scientifically_scored(self):
        def fitter(training, model, rival, spec):
            result = self.fit(model, (1, 4))
            result["converged"] = False
            return result
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, _ = self.execute(Path(temporary), fitter=fitter)
            self.assertFalse(result["rival_diagnostics_complete"])
            self.assertEqual(len(result["fits"]), 1)
            self.assertEqual(result["completed_rivals"], [])
            self.assertEqual(result["diagnostics"], {})
            self.assertEqual(calls.call_count, 1)
            self.assertNotIn("counts_as_curtailment_failure", result)

    def test_nonfinite_heldout_score_refuses_instead_of_dropping_trials(self):
        fit = self.fit("MC", (1, 4))
        scoring = {person: {"rows": rows[:128]} for person, rows in self.people.items()}
        values = np.zeros(128)
        values[5] = np.nan
        with patch.object(core, "trial_likelihood", return_value=values), self.assertRaisesRegex(ValueError, "128 finite"):
            core.score_fit(fit, scoring, self.spec)

    def test_finalization_deadline_retains_diagnostics_without_a_success_publication(self):
        expired = [False]
        publications = []
        original_write = core.write_report
        def alarm(seconds):
            if seconds == 0:
                expired[0] = True
            return 0
        def save(output, report):
            publications.append((report["rival_diagnostics_complete"], "finished_utc" in report))
            return original_write(output, report)
        with tempfile.TemporaryDirectory() as temporary, patch.object(core.signal, "alarm", side_effect=alarm), \
                patch.object(core.time, "monotonic", side_effect=lambda: 1000. if expired[0] else 0.), \
                patch.object(core, "write_report", side_effect=save):
            result, _, _ = self.execute(Path(temporary))
        self.assertEqual(result["status"], "technical_rival_failure")
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertEqual(result["completed_rivals"], ["MC", "MV"])
        self.assertEqual(len(result["fits"]), 4)
        self.assertTrue(result["finished_utc"])
        self.assertFalse(any(complete for complete, _ in publications))


if __name__ == "__main__":
    unittest.main()
