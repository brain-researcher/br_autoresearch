"""Checkpoint identity and once-only scoring tests; no expensive fitting."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import resume_criterion_recovery as resume


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((CODE / "behavioral_implementation.json").read_text())
        self.people = [f"synthetic-{i:03}" for i in range(108)]
        self.versions = {p: "EI" if i < 54 else "ER" for i, p in enumerate(self.people)}

    def fit(self, index):
        model, training, scoring = resume.FIT_KEYS[index]
        names = resume.parameter_names(model, "fixed")
        theta = [self.spec["parameters"][name]["population_center"] for name in names]
        # Different model/half sentinels expose accidental omission or reuse.
        theta[0] = [.2, .6, .4, .8][index]
        return dict(model=model, rival="fixed", parameter_names=names,
                    participants=self.people, parameters=[theta[:] for _ in self.people],
                    global_mean=theta[:], version_means=[theta[:], theta[:]],
                    version_standard_deviations=[[.3]*len(names), [.3]*len(names)],
                    version_order=["EI", "ER"], objective_history=[10000., 10000.00001],
                    converged=True, subject_optimizer_failures=[], support_contacts=[],
                    inference=self.spec["method"], training_runs=list(training),
                    scoring_runs=list(scoring), elapsed_seconds=1.)

    def report(self, count=2):
        return dict(resume.IDENTITY, implementation=copy.deepcopy(self.spec), replicate_index=0,
                    planned_cell_replicates=200, seed=710031, fits=[self.fit(i) for i in range(count)],
                    simulation_seconds=1., simulated_deadline_misses=0, job_id="old-job")

    def simulation(self, directory):
        rows = []
        for person in self.people:
            for run in range(1, 5):
                for trial in range(1, 65):
                    rows.append(dict(participant=person, task_version=self.versions[person], run=run,
                                     trial_order=trial, onset=trial*5, gain=10, loss=10, choice=1,
                                     response_time=1., no_response=0, strength=-1,
                                     response_category="simulated_accept"))
        path = directory / "simulated_events.csv.gz"
        with gzip.open(path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        truth = {person: {name: self.spec["parameters"][name]["population_center"]
                          for name in ("kappa", "log_a", "nondecision_logit", "criterion")}
                 for person in self.people}
        (directory / "generating_parameters.json").write_text(json.dumps(truth))
        return path, truth, rows

    def test_complete_prefix_is_reusable(self):
        report = self.report()
        resume.validate_report_identity(report, self.spec, 0)
        self.assertEqual(list(resume.collect_reusable_fits(report, self.spec, self.people)), resume.FIT_KEYS[:2])

    def test_specification_and_seed_mismatch_are_rejected(self):
        for field, value in [("seed", 710032), ("replicate_index", 1), ("generating_model", "M0")]:
            report = self.report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.validate_report_identity(report, self.spec, 0)
        report = self.report()
        report["implementation"]["optimizer"]["maximum_sweeps"] = 200
        with self.assertRaisesRegex(ValueError, "implementation"):
            resume.validate_report_identity(report, self.spec, 0)

    def test_nonconvergence_or_inconsistent_success_is_rejected(self):
        for field, value in [("converged", False), ("subject_optimizer_failures", [{"participant": self.people[0]}]),
                             ("objective_history", [10000., 9999.])]:
            fit = self.fit(0)
            fit[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.validate_fit(fit, self.spec, self.people)

    def test_duplicate_wrong_half_or_nonprefix_is_rejected(self):
        reports = [self.report(1), self.report(1), self.report(1)]
        reports[0]["fits"].append(copy.deepcopy(reports[0]["fits"][0]))
        reports[1]["fits"][0]["scoring_runs"] = [1, 4]
        reports[2]["fits"] = [self.fit(2)]
        for report in reports:
            with self.subTest(fits=report["fits"][0]["training_runs"]), self.assertRaises(ValueError):
                resume.collect_reusable_fits(report, self.spec, self.people)

    def test_wrong_participant_parameter_identity_and_nan_are_rejected(self):
        for modification in ("participant", "names", "nan", "support"):
            fit = self.fit(0)
            if modification == "participant":
                fit["participants"] = self.people[:-1] + [self.people[0]]
            elif modification == "names":
                fit["parameter_names"] = list(reversed(fit["parameter_names"]))
            else:
                fit["parameters"][0][0] = float("nan") if modification == "nan" else 100.
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                resume.validate_fit(fit, self.spec, self.people)

    def test_duplicate_synthetic_trial_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, truth, rows = self.simulation(directory)
            self.assertEqual(resume.validate_simulation(path, truth, self.report(), self.spec), self.versions)
            rows[-1] = dict(rows[-2])
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                resume.validate_simulation(path, truth, self.report(), self.spec)

    def test_only_missing_fit_runs_and_saved_scores_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old"
            source.mkdir()
            path, _, _ = self.simulation(source)
            original_bytes = path.read_bytes()
            report = self.report(3)
            report["participant_scores"] = {"ignored": "stale_partial_scores"}
            prior_text = json.dumps(report)
            (source / "criterion_recovery.json").write_text(prior_text)
            output = Path(temporary) / "new"
            args = argparse.Namespace(source_attempt=source, output_dir=output,
                                      implementation=CODE / "behavioral_implementation.json", replicate_index=0)
            with patch.object(resume, "require_compute"), patch.dict(resume.os.environ, {"SLURM_JOB_ID": "test-job"}), \
                    patch.object(resume, "fit_partially_pooled", return_value=self.fit(3)) as fitter, \
                    patch.object(resume, "trial_likelihood", side_effect=lambda theta, names, data, spec, rival: np.full(len(data["choice"]), theta[0])), \
                    patch.object(resume, "summarize_paired_scores", return_value={"full_recovery_pass": None}):
                completed = resume.resume(args)
            fitter.assert_called_once()
            self.assertEqual(fitter.call_args.args[1:3], ("MC", "fixed"))
            self.assertEqual({int(row["run"]) for row in fitter.call_args.args[0][self.people[0]]["rows"]}, {2, 3})
            self.assertEqual(completed["reused_fit_count"], 3)
            self.assertEqual(completed["new_fit_count"], 1)
            for scores in completed["participant_scores"].values():
                self.assertAlmostEqual(scores["M0"], .3)
                self.assertAlmostEqual(scores["MC"], .7)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual((output / path.name).read_bytes(), original_bytes)
            self.assertEqual((source / "criterion_recovery.json").read_text(), prior_text)
            with self.assertRaises(FileExistsError), patch.object(resume, "require_compute"):
                resume.resume(args)


if __name__ == "__main__":
    unittest.main()
