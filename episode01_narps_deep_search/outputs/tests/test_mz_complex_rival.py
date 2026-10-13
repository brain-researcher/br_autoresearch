"""Fixed-model extension checks; reuse the already-qualified source fixture."""
import argparse
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
import run_mz_complex_rival as core
from hierarchical_behavior import parameter_names
import test_single_process_rivals as fixtures


class ComplexRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SingleProcessRivalsTests.setUpClass()
        cls.builder = fixtures.SingleProcessRivalsTests()
        cls.spec = cls.builder.spec
        cls.people = cls.builder.people

    def fake_fit(self, training, model, rival, spec):
        self.assertIn(model, ("MVZ", "MVC"))
        self.assertEqual(rival, "fixed")
        self.assertEqual(spec, self.spec)
        self.assertEqual(set(training), set(self.people))
        self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
        runs = tuple(sorted({int(row["run"]) for row in training[next(iter(training))]["rows"]}))
        self.assertIn(runs, ((1, 4), (2, 3)))
        names = parameter_names(model, "fixed")
        self.assertEqual(names, ["kappa", "log_a", "nondecision_logit", "eta", "rho" if model == "MVZ" else "criterion"])
        theta = [spec["parameters"][name]["population_center"] for name in names]
        theta[0] = .5 if runs == (1, 4) else .7
        return {"model": model, "rival": rival, "participants": sorted(training), "parameter_names": names,
            "parameters": [theta[:] for _ in training], "inference": spec["method"], "version_order": ["EI", "ER"],
            "global_mean": theta[:], "version_means": [theta[:], theta[:]],
            "version_standard_deviations": [[.3] * len(names), [.3] * len(names)],
            "converged": True, "subject_optimizer_failures": [], "support_contacts": [],
            "objective_history": [10000., 10000.00001]}

    def execute(self, directory, model, fitter=None):
        source = directory / "source"
        self.builder.fixture(source)
        args = argparse.Namespace(model=model, source_dir=source, implementation=core.IMPLEMENTATION,
                                  output_dir=directory / "rival", max_seconds=60)
        original_write = core.write_report
        publications = []
        def write(output, report):
            publications.append((report["rival_diagnostics_complete"], "finished_utc" in report))
            return original_write(output, report)
        with patch.object(core, "SOURCE", source), patch.object(core, "require_compute"), \
                patch.dict(core.os.environ, fixtures.THREADS), patch.object(core, "write_report", side_effect=write), \
                patch.object(core.shared, "trial_likelihood", side_effect=self.builder.likelihood), \
                patch.object(core, "fit_partially_pooled", side_effect=fitter or self.fake_fit) as fits:
            result = core.run(args)
            with self.assertRaises(FileExistsError):
                core.run(args)
        return result, fits, args, publications

    def test_each_model_fits_only_its_two_directions_and_preserves_saved_baselines(self):
        for model in core.MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                result, fits, args, publications = self.execute(Path(temporary), model)
                self.assertEqual([call.args[1] for call in fits.call_args_list], [model, model])
                self.assertEqual([fit["training_runs"] for fit in result["fits"]], [[1, 4], [2, 3]])
                self.assertEqual([fit["scoring_runs"] for fit in result["fits"]], [[2, 3], [1, 4]])
                self.assertTrue(result["rival_diagnostics_complete"])
                self.assertEqual(result["completed_rivals"], [model])
                self.assertEqual(len(result["scored_primary_directions"]), 2)
                for row in result["participant_scores"].values():
                    self.assertEqual(set(row), {"M0", "MZ", model, "version"})
                    self.assertAlmostEqual(row["M0"], .3)
                    self.assertAlmostEqual(row["MZ"], .7)
                    self.assertAlmostEqual(row[model], .6)
                self.assertEqual(result["missing_nested_model_comparisons"], ["MV"] if model == "MVZ" else ["MV", "MC"])
                for field in ("family_selection", "confusion_rate", "group_recovery", "full_gate"):
                    self.assertEqual(result[field], "not_evaluated")
                self.assertFalse(result["simulation_regenerated"])
                self.assertFalse(result["saved_M0_MZ_refitted"])
                self.assertNotIn("counts_as_curtailment_failure", result)
                self.assertEqual(publications[-1], (True, True))
                self.assertFalse(any(complete for complete, _ in publications[:-1]))
                self.assertEqual(json.loads((args.output_dir / core.REPORT_NAME).read_text()), result)

    def test_descriptive_paired_sign_SE_and_nested_status_for_both_models(self):
        effects = np.linspace(-.2, .4, 108)
        for model in core.MODELS:
            scores = {person: {"M0": -1., "MZ": -.9, model: -.9 + effect, "version": rows[0]["task_version"]}
                      for (person, rows), effect in zip(self.people.items(), effects)}
            result = core.paired_diagnostics(scores, model)
            mz = result["model_minus_saved_MZ"]
            self.assertEqual(mz["contrast"], f"{model}-MZ")
            self.assertAlmostEqual(mz["mean_gain_nats_per_trial"], float(effects.mean()))
            self.assertAlmostEqual(mz["participant_SE"], float(effects.std(ddof=1) / np.sqrt(108)))
            self.assertAlmostEqual(mz["version_means"]["ER"], float(effects[54:].mean()))
            self.assertEqual(mz["nested_comparison"], model == "MVZ")
            self.assertTrue(result["model_minus_saved_M0"]["nested_comparison"])
            self.assertAlmostEqual(result["model_minus_saved_M0"]["mean_gain_nats_per_trial"], float(effects.mean()) + .1)
            scores["synthetic-000"][model] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                core.paired_diagnostics(scores, model)

    def test_second_fit_timeout_preserves_first_fit_without_incomplete_full_scores(self):
        def fitter(training, model, rival, spec):
            runs = {int(row["run"]) for row in training[next(iter(training))]["rows"]}
            if runs == {2, 3}:
                core.interrupted(signal.SIGTERM, None)
            return self.fake_fit(training, model, rival, spec)
        with tempfile.TemporaryDirectory() as temporary:
            result, fits, args, publications = self.execute(Path(temporary), "MVC", fitter)
            self.assertEqual(result["status"], "technical_rival_failure")
            self.assertFalse(result["rival_diagnostics_complete"])
            self.assertEqual(len(result["fits"]), 1)
            self.assertEqual(len(result["scored_primary_directions"]), 1)
            self.assertEqual(result["diagnostics"], {})
            self.assertNotIn("participant_scores", result)
            self.assertEqual(fits.call_count, 2)
            self.assertFalse(any(complete for complete, _ in publications))
            self.assertTrue(result["finished_utc"])
            self.assertEqual(json.loads((args.output_dir / core.REPORT_NAME).read_text()), result)

    def test_unselected_model_source_deadline_and_threads_refuse_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "never_created"
            original = dict(model="MVZ", source_dir=core.SOURCE, implementation=core.IMPLEMENTATION,
                            output_dir=destination, max_seconds=5400)
            with patch.object(core, "require_compute"), patch.dict(core.os.environ, fixtures.THREADS):
                for change in ({"model": "MZC"}, {"source_dir": Path(temporary)}, {"max_seconds": 5401}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        core.run(argparse.Namespace(**(original | change)))
                with patch.dict(core.os.environ, {"OMP_NUM_THREADS": "2"}), self.assertRaisesRegex(ValueError, "thread"):
                    core.run(argparse.Namespace(**original))
            self.assertFalse(destination.exists())

    def test_late_finalization_timeout_preserves_two_fits_but_never_publishes_success(self):
        expired = [False]
        def alarm(seconds):
            if seconds == 0:
                expired[0] = True
            return 0
        with tempfile.TemporaryDirectory() as temporary, patch.object(core.signal, "alarm", side_effect=alarm), \
                patch.object(core.time, "monotonic", side_effect=lambda: 1000. if expired[0] else 0.):
            result, _, _, publications = self.execute(Path(temporary), "MVZ")
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertEqual(result["status"], "technical_rival_failure")
        self.assertEqual(len(result["fits"]), 2)
        self.assertEqual(result["completed_rivals"], ["MVZ"])
        self.assertIn("MVZ", result["diagnostics"])
        self.assertFalse(any(complete for complete, _ in publications))
        self.assertTrue(result["finished_utc"])


if __name__ == "__main__":
    unittest.main()
