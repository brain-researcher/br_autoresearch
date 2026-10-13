"""Closed-source recovery-cell checks; no expensive fit or recovery experiment."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_single_process_recovery as core


def cohort():
    result = {}
    for index in range(108):
        person, version = f"synthetic-{index:03}", "EI" if index < 54 else "ER"
        rows = []
        for offer in range(256):
            rows.append(dict(participant=person, task_version=version, run=str(offer // 64 + 1),
                trial_order=str(offer % 64 + 1), onset=str((offer % 64) * 5 + 1),
                gain=str(10 + 2 * (offer // 16) if version == "EI" else 5 + offer // 16),
                loss=str(5 + offer % 16)))
        result[person] = rows
    return result


def deterministic_responses(rng, drift, a, z, t0, eps, miss, *, deadline):
    choices = np.resize(np.array([-1, 0, 1]), len(drift))
    return choices, np.where(choices == -1, np.nan, 1.)


class SingleProcessRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())
        cls.source = cohort()

    def write_design(self, directory, rows=None):
        path = directory / "old_simulated_events.csv.gz"
        rows = rows if rows is not None else [r for person in self.source.values() for r in person]
        with gzip.open(path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=core.DESIGN_FIELDS + ("choice", "response_time", "sequence_template"))
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, choice="old-response-must-not-be-parsed", response_time="not-an-RT",
                                     sequence_template="forbidden-extra-design-column"))
        return path

    def test_exact_focal_truth_and_disjoint_fixed_seed_banks(self):
        self.assertEqual(core.focal_truth("MV"), {"eta": math.log(1.15), "rho": 0., "criterion": 0.})
        self.assertEqual(core.focal_truth("MZ"), {"eta": 0., "rho": math.log(.45/.55), "criterion": 0.})
        self.assertAlmostEqual(expit(core.focal_truth("MZ")["rho"]), .45)
        self.assertEqual(core.cell_seed("MV", 0), 810031)
        self.assertEqual(core.cell_seed("MZ", 199), 910031 + 10007 * 199)
        self.assertFalse({core.cell_seed("MV", i) for i in range(200)} &
                         {core.cell_seed("MZ", i) for i in range(200)})
        for model, index in (("MC", 0), ("MV", -1), ("MZ", 200), ("MV", True)):
            with self.subTest(model=model, index=index), self.assertRaises(ValueError):
                core.cell_seed(model, index)

    def test_projection_never_indexes_unapproved_columns(self):
        header = ("choice",) + core.DESIGN_FIELDS + ("response_time", "sequence_template")
        expected = self.source["synthetic-000"][0]
        class GuardedValues:
            def __getitem__(self, index):
                if header[index] not in core.DESIGN_FIELDS:
                    raise AssertionError("A non-design value was accessed")
                return expected[header[index]]
        self.assertEqual(core.project_design(header, GuardedValues()), expected)
        with self.assertRaises(ValueError):
            core.project_design(header + ("gain",), GuardedValues())

    def test_exact_design_support_projection_and_source_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_design(directory)
            with patch.object(core, "DESIGN_SOURCE", path):
                self.assertEqual(core.load_design(path), self.source)
                with self.assertRaises(ValueError):
                    core.load_design(directory / "observed_events.csv.gz")
                rows = [dict(r) for people in self.source.values() for r in people]
                rows[-1] = dict(rows[-2])
                self.write_design(directory, rows)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    core.load_design(path)

    def test_version_balance_and_exact_offer_matrix_are_not_count_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [dict(r) for people in self.source.values() for r in people]
            for row in rows:
                if row["participant"] == "synthetic-054":
                    row["task_version"] = "EI"
            path = self.write_design(directory, rows)
            with patch.object(core, "DESIGN_SOURCE", path), self.assertRaisesRegex(ValueError, "support"):
                core.load_design(path)
            rows = [dict(r) for people in self.source.values() for r in people]
            rows[1]["loss"] = rows[0]["loss"]
            path = self.write_design(directory, rows)
            with patch.object(core, "DESIGN_SOURCE", path), self.assertRaisesRegex(ValueError, "offer matrix"):
                core.load_design(path)

    def test_fresh_simulation_uses_fixed_truth_nuisance_and_likelihood_arguments(self):
        source = {"synthetic-000": self.source["synthetic-000"][:3]}
        for model in ("MV", "MZ"):
            with self.subTest(model=model), patch.object(core.deadline_ddm, "simulate", side_effect=deterministic_responses) as simulator:
                truth, rows = core.simulate_design(source, model, core.cell_seed(model, 0), self.spec)
            rng = np.random.default_rng(core.cell_seed(model, 0))
            expected = {}
            for name in ("kappa", "log_a", "nondecision_logit"):
                prior = self.spec["parameters"][name]
                expected[name] = core.truncated_normal(rng, prior["population_center"], prior["individual_scale_center"], prior["bounds"])
            expected.update(core.focal_truth(model))
            self.assertEqual(truth["synthetic-000"], expected)
            args = simulator.call_args.args
            direct = np.exp(expected["kappa"] - expected["eta"]/2) * np.array([10., 10., 10.])/10
            direct -= np.exp(expected["kappa"] + expected["eta"]/2) * np.array([5., 6., 7.])/10
            np.testing.assert_allclose(args[1], direct, rtol=1e-14)
            np.testing.assert_allclose(args[2:5], [np.exp(expected["log_a"]), expit(expected["rho"]),
                                                  4 * expit(expected["nondecision_logit"])])
            self.assertEqual(args[5:7], (.02, .01))
            self.assertEqual(simulator.call_args.kwargs, {"deadline": 4.})
            self.assertEqual([row["choice"] for row in rows], [-1, 0, 1])
            self.assertEqual([row["response_time"] for row in rows], ["", 1., 1.])
            self.assertEqual([row["response_category"] for row in rows], ["no_response", "simulated_reject", "simulated_accept"])
            self.assertTrue(all(row["strength"] == -1 for row in rows))
            self.assertTrue(all(set(row) == set(core.DESIGN_FIELDS + core.RESPONSE_FIELDS) for row in rows))

    def test_only_core_margin_and_two_SE_can_count_curtailment_failure(self):
        def scores(model, effects):
            return {p: {"M0": -1., model: -1. + effect, "version": rows[0]["task_version"]}
                    for (p, rows), effect in zip(self.source.items(), effects)}
        for model in ("MV", "MZ"):
            precise_small = core.summarize_paired_scores(scores(model, [.005] * 108), model, 3)
            self.assertTrue(precise_small["counts_as_curtailment_failure"])
            imprecise = core.summarize_paired_scores(scores(model, [.011 + (.1 if i % 2 else -.1) for i in range(108)]), model, 3)
            self.assertTrue(imprecise["counts_as_curtailment_failure"])
            diagnostic = core.summarize_paired_scores(scores(model, [.05] * 54 + [-.001] * 54), model, 3)
            self.assertTrue(diagnostic["core_family_selection_pass"])
            self.assertFalse(diagnostic["counts_as_curtailment_failure"])
            self.assertFalse(diagnostic["richer_primary_diagnostic_pass"])
            self.assertIsNone(diagnostic["full_recovery_pass"])
        invalid = scores("MV", [.02] * 108)
        invalid["synthetic-000"]["MV"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            core.summarize_paired_scores(invalid, "MV", 3)

    def fake_fit(self, training, model, rival, spec):
        runs = sorted({int(row["run"]) for row in training["synthetic-000"]["rows"]})
        self.assertIn(runs, ([1, 4], [2, 3]))
        self.assertTrue(all(len(person["rows"]) == 128 for person in training.values()))
        self.assertEqual(rival, "fixed")
        self.assertEqual(spec, self.spec)
        names = core.parameter_names(model, rival)
        theta = [spec["parameters"][name]["population_center"] for name in names]
        theta[0] = {("M0", (1, 4)): .2, ("M0", (2, 3)): .4}.get((model, tuple(runs)), .6 if runs == [1, 4] else .8)
        return {"model": model, "rival": rival, "participants": sorted(training), "parameter_names": names,
                "parameters": [theta[:] for _ in training], "inference": spec["method"], "version_order": ["EI", "ER"],
                "global_mean": theta[:], "version_means": [theta[:], theta[:]],
                "version_standard_deviations": [[.3] * len(names), [.3] * len(names)],
                "converged": True, "subject_optimizer_failures": [], "support_contacts": [],
                "objective_history": [10000., 10000.00001]}

    def execute_mocked(self, directory, model="MV", fitter=None, likelihood=None):
        source = directory / "old"
        source.mkdir()
        path = self.write_design(source)
        args = argparse.Namespace(model=model, replicate_index=0, design_events=path,
            implementation=core.IMPLEMENTATION, output_dir=directory / "fresh", max_seconds=60)
        with patch.object(core, "DESIGN_SOURCE", path), patch.object(core, "require_compute"), \
                patch.object(core.deadline_ddm, "simulate", side_effect=deterministic_responses), \
                patch.object(core, "fit_partially_pooled", side_effect=fitter or self.fake_fit) as fit_call, \
                patch.object(core, "trial_likelihood", side_effect=likelihood or
                    (lambda theta, names, data, spec, rival: np.full(len(data["choice"]), theta[0]))):
            report = core.run(args)
            with self.assertRaises(FileExistsError):
                core.run(args)
        return report, fit_call, args

    def test_four_fixed_fits_disjoint_halves_once_only_scoring_and_checkpoints(self):
        for model in ("MV", "MZ"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                report, fitter, args = self.execute_mocked(Path(temporary), model)
                self.assertEqual([call.args[1] for call in fitter.call_args_list], ["M0", model, "M0", model])
                self.assertEqual(len(report["fits"]), 4)
                self.assertTrue(report["valid_recovery_component"])
                self.assertEqual(report["full_behavioral_gate"], "not_evaluated")
                for scores in report["participant_scores"].values():
                    self.assertAlmostEqual(scores["M0"], .3)
                    self.assertAlmostEqual(scores[model], .7)
                saved = json.loads((args.output_dir / core.REPORT_NAME).read_text())
                self.assertEqual(saved, report)
                truth = json.loads((args.output_dir / "generating_parameters.json").read_text())
                self.assertEqual(len(truth), 108)
                self.assertTrue(all(p["criterion"] == 0. for p in truth.values()))

    def test_nonconvergence_nonfinite_scores_and_signals_are_technical(self):
        def bad_fit(*args):
            fit = self.fake_fit(*args)
            fit["converged"] = False
            return fit
        def terminate(*args):
            core.interrupted(signal.SIGTERM, None)
        for kind in ("nonconverged", "nonfinite", "terminated"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                report, _, args = self.execute_mocked(Path(temporary),
                    fitter=bad_fit if kind == "nonconverged" else terminate if kind == "terminated" else None,
                    likelihood=(lambda *args: np.full(128, np.nan)) if kind == "nonfinite" else None)
                self.assertEqual(report["status"], "technical_recovery_failure")
                self.assertFalse(report["valid_recovery_component"])
                self.assertFalse(report["counts_as_curtailment_failure"])
                self.assertNotIn("comparison", report)
                self.assertTrue((args.output_dir / "simulated_events.csv.gz").exists())
                self.assertEqual(len(report["fits"]), 0 if kind == "terminated" else 1)


if __name__ == "__main__":
    unittest.main()
