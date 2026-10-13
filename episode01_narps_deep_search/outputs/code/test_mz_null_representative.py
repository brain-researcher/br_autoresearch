"""Focused null-identity, claim-boundary, and checkpoint tests; run on compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hierarchical_behavior import parameter_names
import run_mz_null_representative as null


THREADS = {name: "1" for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def row(person="p000", version="EI", gain=10, loss=10, trial=1):
    return {"participant": person, "task_version": version, "run": str((trial - 1) // 64 + 1),
            "trial_order": str((trial - 1) % 64 + 1), "onset": str(trial * 5),
            "gain": str(gain), "loss": str(loss)}


def full_design():
    source = {}
    for index in range(108):
        person, version = f"p{index:03d}", "EI" if index < 54 else "ER"
        gains = range(10, 41, 2) if version == "EI" else range(5, 21)
        source[person] = [row(person, version, gain, loss, trial)
                          for trial, (gain, loss) in enumerate(
                              ((gain, loss) for gain in gains for loss in range(5, 21)), 1)]
    return source


class NullRepresentativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(null.IMPLEMENTATION.read_text())

    def arguments(self, output, **changes):
        values = dict(design_events=null.DESIGN_SOURCE, implementation=null.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_null_identity_frozen_seed_and_fresh_censoring(self):
        old = row()
        old.update(choice=0, response_time=2.7, response_category="old_response", private_extra="unused")
        source = {"p000": [old, row(gain=20, loss=10, trial=2)]}
        expected_rng = np.random.default_rng(1300037)
        expected = {}
        for name in ("kappa", "log_a", "nondecision_logit"):
            prior = self.spec["parameters"][name]
            expected[name] = null.truncated_normal(expected_rng, prior["population_center"],
                                                    prior["individual_scale_center"], prior["bounds"])
        with patch.object(null.deadline_ddm, "simulate", return_value=(np.array([-1, 1]), np.array([np.nan, .4]))) as simulator:
            truth, simulated = null.simulate_null(source, self.spec)
        self.assertEqual(truth["p000"], {**expected, "eta": 0., "rho": 0., "criterion": 0.})
        args = simulator.call_args.args
        np.testing.assert_allclose(args[1], [0., math.exp(expected["kappa"])], rtol=1e-14)
        self.assertEqual(args[3], .5)
        self.assertEqual(args[5:7], (.02, .01))
        self.assertEqual(simulator.call_args.kwargs, {"deadline": 4.})
        self.assertEqual(set(simulated[0]), set(null.DESIGN_FIELDS + null.RESPONSE_FIELDS))
        self.assertEqual((simulated[0]["choice"], simulated[0]["no_response"], simulated[0]["response_time"]), (-1, 1, ""))
        self.assertEqual(simulated[1]["response_category"], "simulated_accept")
        self.assertEqual(simulated[1]["response_time"], .4)

    def test_invalid_generated_deadline_or_response_is_rejected(self):
        for choices, times in (([-1], [1.]), ([1], [4.01]), ([1], [np.nan])):
            with self.subTest(choices=choices, times=times), patch.object(
                    null.deadline_ddm, "simulate", return_value=(np.array(choices), np.array(times))):
                with self.assertRaises(ValueError):
                    null.simulate_null({"p000": [row()]}, self.spec)

    def test_scope_and_time_limits_fail_before_loading_design(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(null, "require_compute"), \
                patch.dict(os.environ, THREADS), patch.object(null, "load_design") as loader:
            for changes in ({"design_events": Path(directory) / "other.csv.gz"},
                            {"implementation": Path(directory) / "other.json"},
                            {"max_seconds": 0}, {"max_seconds": 5401}, {"max_seconds": True}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    null.run(self.arguments(Path(directory) / "fresh", **changes))
            loader.assert_not_called()
            self.assertFalse((Path(directory) / "fresh").exists())

    def test_existing_attempt_and_multithreading_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(null, "require_compute"), \
                patch.dict(os.environ, THREADS), patch.object(null, "load_design") as loader:
            existing = Path(directory) / "existing"
            existing.mkdir()
            marker = existing / "preserve.txt"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                null.run(self.arguments(existing))
            with patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}), self.assertRaises(ValueError):
                null.run(self.arguments(Path(directory) / "fresh"))
            self.assertEqual(marker.read_text(), "original")
            loader.assert_not_called()

    def run_mocked_fits(self, output, *, reject_second=False):
        """Exercise real orchestration/fit validation with a bounded synthetic fit stub."""
        source = full_design()
        calls = []

        def fitter(training, model, rival, spec):
            calls.append(model)
            names = parameter_names(model, rival)
            centers = [spec["parameters"][name]["population_center"] for name in names]
            scales = [spec["parameters"][name]["individual_scale_center"] for name in names]
            return {"model": model, "rival": rival, "participants": sorted(source), "parameter_names": names,
                    "version_order": ["EI", "ER"], "inference": spec["method"],
                    "converged": not (reject_second and len(calls) == 2), "subject_optimizer_failures": [],
                    "parameters": [list(centers) for _ in source], "global_mean": centers,
                    "version_means": [list(centers), list(centers)],
                    "version_standard_deviations": [list(scales), list(scales)],
                    "objective_history": [100., 100.], "support_contacts": []}

        def simulator(rng, drift, *args, **kwargs):
            return np.ones(len(drift), dtype=int), np.full(len(drift), .5)

        def scorer(fit, scoring, spec, check):
            check()
            return {person: -1. if fit["model"] == "M0" else -.99 for person in source}

        with ExitStack() as stack:
            stack.enter_context(patch.object(null, "require_compute"))
            stack.enter_context(patch.dict(os.environ, THREADS))
            stack.enter_context(patch.object(null, "load_design", return_value=source))
            stack.enter_context(patch.object(null.deadline_ddm, "simulate", side_effect=simulator))
            stack.enter_context(patch.object(null, "load_split", return_value=(source, source)))
            stack.enter_context(patch.object(null, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(null, "score_fit", side_effect=scorer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = null.run(self.arguments(output))
        return result, calls

    def test_four_fits_preserve_null_and_do_not_select_a_family_or_wave(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "representative"
            report, calls = self.run_mocked_fits(output)
            self.assertEqual(calls, ["M0", "MZ", "M0", "MZ"])
            self.assertEqual(report["generating_model"], "M0")
            self.assertEqual((report["seed"], report["replicate_index"], report["SESOI_level"]), (1300037, 0, 0.))
            self.assertTrue(report["pairwise_diagnostic_complete"])
            self.assertTrue(report["nested_comparison"]["qualifies_nested_comparison"])
            self.assertAlmostEqual(report["nested_comparison"]["mean_gain_nats_per_trial"], .02)
            self.assertEqual([(fit["model"], fit["training_runs"], fit["scoring_runs"]) for fit in report["fits"]],
                             [(model, list(train), list(held)) for train, held in null.HALVES for model in ("M0", "MZ")])
            for field in ("family_selection", "full_menu_false_selection", "false_selection_rate",
                          "confusion_rate", "group_recovery", "interval_coverage", "full_gate"):
                self.assertEqual(report[field], "not_evaluated")
            self.assertIsNone(report["full_recovery_pass"])
            self.assertFalse(report["episode_terminal"])
            self.assertFalse(report["wave_selected"])
            self.assertNotIn("counts_as_curtailment_failure", report)
            self.assertNotIn("core_family_selection_pass", report)
            saved = json.loads((output / null.REPORT_NAME).read_text())
            self.assertEqual(saved, report)
            truth = json.loads((output / "generating_parameters.json").read_text())
            self.assertEqual(len(truth), 108)
            self.assertTrue(all(all(values[key] == 0. for key in ("eta", "rho", "criterion")) for values in truth.values()))
            with gzip.open(output / "simulated_events.csv.gz", "rt", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(null.DESIGN_FIELDS + null.RESPONSE_FIELDS))
                self.assertEqual(sum(1 for _ in reader), 27648)

    def test_nonconvergence_retains_returned_fits_but_releases_no_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "technical"
            report, calls = self.run_mocked_fits(output, reject_second=True)
            self.assertEqual(calls, ["M0", "MZ"])
            self.assertEqual(report["status"], "technical_representative_failure")
            self.assertFalse(report["pairwise_diagnostic_complete"])
            self.assertEqual(len(report["fits"]), 2)
            self.assertTrue(report["fits"][0]["converged"])
            self.assertFalse(report["fits"][1]["converged"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_comparison", report)
            self.assertNotIn("counts_as_curtailment_failure", report)
            self.assertTrue((output / "simulated_events.csv.gz").exists())
            self.assertTrue((output / "generating_parameters.json").exists())
            self.assertEqual(json.loads((output / null.REPORT_NAME).read_text())["fits"], report["fits"])


if __name__ == "__main__":
    unittest.main()
