"""Paired nested-model arithmetic and explicit, completed-panel boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import summarize_mz_fixed_menu as core
from hierarchical_behavior import parameter_names
import test_single_process_rivals as fixtures


class FixedMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SingleProcessRivalsTests.setUpClass()
        cls.builder = fixtures.SingleProcessRivalsTests()
        cls.spec = copy.deepcopy(cls.builder.spec)
        with tempfile.TemporaryDirectory() as temporary:
            cls.base, _, _ = cls.builder.fixture(Path(temporary) / "synthetic_source")
        cls.simple = cls.addon(("MC", "MV"))
        cls.mvz = cls.addon(("MVZ",))
        cls.mvc = cls.addon(("MVC",))

    @classmethod
    def addon(cls, models):
        complex_model = models[0] if len(models) == 1 else None
        score_values = {"MC": .6, "MV": 1., "MVZ": 1.2, "MVC": 1.1}
        fits = []
        for model in models:
            for train, held in core.HALVES:
                fit = copy.deepcopy(cls.builder.fit("MZ", train))
                names = parameter_names(model, "fixed")
                theta = [cls.spec["parameters"][name]["population_center"] for name in names]
                theta[0] = score_values[model] + (-.1 if train == (1, 4) else .1)
                fit.update(model=model, parameter_names=names, parameters=[theta[:] for _ in range(108)],
                           global_mean=theta[:], version_means=[theta[:], theta[:]],
                           version_standard_deviations=[[.3] * len(names), [.3] * len(names)],
                           training_runs=list(train), scoring_runs=list(held))
                fits.append(fit)
        scores = {person: {**row, **{model: score_values[model] for model in models}}
                  for person, row in cls.base["participant_scores"].items()}
        report = {"analysis_id": "behavioral_mechanism_kill_test",
            "phase": "MZ0_complex_rival_diagnostics" if complex_model else "MZ0_MC_MV_rival_diagnostics",
            "generating_model": "MZ", "replicate_index": 0, "seed": 910031,
            "implementation": copy.deepcopy(cls.spec), "source_dir": str(core.SOURCE),
            "source_finished_utc": cls.base["finished_utc"],
            "started_utc": "2026-10-05T00:02:00+00:00", "finished_utc": "2026-10-05T00:03:00+00:00",
            "participants": 108, "runs": 432, "trials": 27648,
            "status": "rival_diagnostics_completed", "rival_diagnostics_complete": True,
            "saved_scores_replayed": True, "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
            "completed_rivals": list(models), "family_selection": "not_evaluated",
            "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated", "full_gate": "not_evaluated",
            "full_recovery_pass": None, "episode_terminal": False, "fits": fits, "participant_scores": scores}
        if complex_model:
            report.update(fitted_model=complex_model, scored_primary_directions=[
                {"training_runs": list(train), "scoring_runs": list(held)} for train, held in core.HALVES])
        return report

    def panels(self):
        return copy.deepcopy((self.base, self.simple, self.mvz, self.mvc))

    def test_exact_nine_nested_edges_and_paired_means(self):
        result = core.build_summary(*self.panels(), self.spec)
        expected = ("MV-M0", "MZ-M0", "MC-M0", "MVZ-M0", "MVZ-MV", "MVZ-MZ",
                    "MVC-M0", "MVC-MV", "MVC-MC")
        self.assertEqual(tuple(result["nested_comparisons"]), expected)
        self.assertNotIn("MVC-MZ", result["nested_comparisons"])
        self.assertNotIn("MZ-MC", result["nested_comparisons"])
        known = {"M0": .3, "MV": 1., "MZ": .7, "MC": .6, "MVZ": 1.2, "MVC": 1.1}
        for name, contrast in result["nested_comparisons"].items():
            model, baseline = name.split("-")
            self.assertAlmostEqual(contrast["mean_gain_nats_per_trial"], known[model] - known[baseline])
            self.assertAlmostEqual(contrast["participant_SE"], 0.)
            self.assertEqual(contrast["participants"], 108)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MVZ", "MVC"])
        self.assertTrue(result["multiple_incomparable_eligible_models"])
        self.assertEqual(result["family_selection"], "not_evaluated")

    def test_mean_margin_and_paired_SE_are_independent_requirements(self):
        def panel(values):
            return {person: {"M0": 0., "MV": float(value), "version": row["version"]}
                    for (person, row), value in zip(self.base["participant_scores"].items(), values)}
        small = core.nested_contrast(panel([.005] * 108), "MV", "M0")
        self.assertFalse(small["requirements"]["mean_gain_at_least_0p01"])
        self.assertTrue(small["requirements"]["mean_gain_greater_than_two_participant_SE"])
        values = .011 + np.tile([-.1, .1], 54)
        imprecise = core.nested_contrast(panel(values), "MV", "M0")
        self.assertTrue(imprecise["requirements"]["mean_gain_at_least_0p01"])
        self.assertFalse(imprecise["requirements"]["mean_gain_greater_than_two_participant_SE"])
        self.assertAlmostEqual(imprecise["participant_SE"], float(values.std(ddof=1) / np.sqrt(108)))
        good = core.nested_contrast(panel([.02] * 108), "MV", "M0")
        self.assertTrue(good["qualifies_nested_comparison"])

    def test_unique_or_incomparable_eligibility_never_selects_a_family(self):
        for mv, mc, expected in ((.305, .302, ["MZ"]), (.32, .32, ["MV", "MZ", "MC"])):
            base, simple, mvz, mvc = self.panels()
            for row in simple["participant_scores"].values():
                row.update(MV=mv, MC=mc)
            for row in mvz["participant_scores"].values():
                row["MVZ"] = .704
            for row in mvc["participant_scores"].values():
                row["MVC"] = .306 if mv < .31 else .325
            result = core.build_summary(base, simple, mvz, mvc, self.spec)
            self.assertEqual(result["maximal_eligible_models_diagnostic"], expected)
            self.assertEqual(result["multiple_incomparable_eligible_models"], len(expected) > 1)
            for field in ("family_selection", "confusion_rate", "group_recovery", "full_gate"):
                self.assertEqual(result[field], "not_evaluated")
            self.assertFalse(result["episode_terminal"])
            self.assertIsNone(result["full_recovery_pass"])
            self.assertNotIn("selected_model", result)
            self.assertNotIn("counts_as_curtailment_failure", result)

    def test_incomplete_wrong_cohort_model_or_implementation_panels_refuse(self):
        for field, value in (("status", "running"), ("rival_diagnostics_complete", False),
                             ("finished_utc", ""), ("source_finished_utc", "another cohort"),
                             ("seed", 910032), ("replicate_index", True),
                             ("source_dir", str(core.SOURCE.parent / "different")),
                             ("simulation_regenerated", True), ("saved_M0_MZ_refitted", True)):
            reports = self.panels()
            reports[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                core.build_summary(*reports, self.spec)
        reports = self.panels()
        reports[2]["implementation"]["optimizer"]["maximum_sweeps"] = 501
        with self.assertRaisesRegex(ValueError, "implementation"):
            core.build_summary(*reports, self.spec)
        reports = self.panels()
        reports[3]["fitted_model"] = "MVZ"
        with self.assertRaisesRegex(ValueError, "Complex-rival"):
            core.build_summary(*reports, self.spec)

    def test_shared_saved_baselines_people_and_versions_cannot_drift(self):
        for change in ("baseline", "people", "versions", "nonfinite"):
            reports = self.panels()
            scores = reports[1]["participant_scores"]
            if change == "baseline":
                scores["synthetic-000"]["MZ"] += 1e-14
            elif change == "people":
                scores["different-person"] = scores.pop("synthetic-000")
            elif change == "versions":
                scores["synthetic-000"]["version"], scores["synthetic-054"]["version"] = "ER", "EI"
            else:
                scores["synthetic-000"]["MC"] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.build_summary(*reports, self.spec)

    def test_missing_nonconverged_or_wrong_half_fit_cannot_be_pooled(self):
        for change in ("missing", "nonconverged", "wrong_half"):
            reports = self.panels()
            if change == "missing":
                reports[2]["fits"].pop()
            elif change == "nonconverged":
                reports[2]["fits"][0]["converged"] = False
            else:
                reports[2]["fits"][0]["training_runs"] = [2, 3]
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                core.build_summary(*reports, self.spec)

    def test_explicit_four_report_cli_writes_fresh_output_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("base.json", "simple.json", "mvz.json", "mvc.json")]
            originals = []
            for path, report in zip(paths, self.panels()):
                payload = json.dumps(report)
                path.write_text(payload)
                originals.append(payload)
            implementation = root / "implementation.json"
            implementation.write_text(json.dumps(self.spec))
            output = root / "summary.json"
            command = ["--base-report", str(paths[0]), "--simple-rivals-report", str(paths[1]),
                       "--mvz-report", str(paths[2]), "--mvc-report", str(paths[3]),
                       "--implementation", str(implementation), "--output", str(output)]
            with patch.object(core, "BASE_REPORT", paths[0]), patch.object(core, "IMPLEMENTATION", implementation), \
                    patch.object(core, "require_compute"), patch("builtins.print"):
                self.assertEqual(core.main(command), 0)
                with self.assertRaises(FileExistsError):
                    core.main(command)
                duplicate = command[:]
                duplicate[duplicate.index("--mvc-report") + 1] = str(paths[2])
                with self.assertRaisesRegex(ValueError, "four distinct"):
                    core.main(duplicate)
            self.assertEqual([path.read_text() for path in paths], originals)
            result = json.loads(output.read_text())
            self.assertEqual(len(result["nested_comparisons"]), 9)
            self.assertEqual(result["family_selection"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
