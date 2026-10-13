"""Saved-index binding/replay tests with authored cohorts and mocked fitting."""
import argparse
from copy import deepcopy
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

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_mz_saved_replicate_rival as core
from test_mz_group_recovery import PEOPLE, report_fixture


THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def fixture_people():
    people = {}
    for index, person in enumerate(PEOPLE):
        version = "EI" if index < 54 else "ER"
        rows = []
        for offer in range(256):
            missing = offer % 64 == 0
            rows.append({"participant": person, "task_version": version,
                "run": str(offer // 64 + 1), "trial_order": str(offer % 64 + 1),
                "onset": str((offer % 64) * 5 + 1),
                "gain": str(10 + 2 * (offer // 16) if version == "EI" else 5 + offer // 16),
                "loss": str(5 + offer % 16), "choice": "-1" if missing else "1",
                "no_response": "1" if missing else "0", "strength": "-1",
                "response_time": "" if missing else "1.25",
                "response_category": "no_response" if missing else "simulated_accept"})
        people[person] = rows
    return people


class SavedReplicateRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())
        cls.people = fixture_people()
        params = {name: cls.spec["parameters"][name]["population_center"] for name in ("kappa", "log_a", "nondecision_logit")}
        params.update(core.focal_truth("MZ"))
        cls.truth = {person: dict(params) for person in PEOPLE}

    def original(self, index=21, *, epsilon=0., core_pass=True):
        report = report_fixture(self.spec, index=index, core_pass=core_pass)
        report.update(started_utc="2026-10-04T23:00:00+00:00", design_source=str(core.DESIGN_SOURCE),
                      design_fields=list(core.DESIGN_FIELDS), simulated_deadline_misses=432)
        for fit in report["fits"]:
            first = fit["training_runs"] == [1, 4]
            kappa = (.2 if first else .4) if fit["model"] == "M0" else (.6 if first else .8)
            for row in fit["parameters"]:
                row[0] = kappa
        report["participant_scores"] = {p: {"version": "EI" if i < 54 else "ER", "M0": .3 + epsilon, "MZ": .7}
                                         for i, p in enumerate(PEOPLE)}
        return report

    def fake_fit(self, training, model, rival, spec):
        self.assertIn(model, core.MODELS)
        self.assertEqual(rival, "fixed")
        self.assertEqual(spec, self.spec)
        self.assertEqual(set(training), set(PEOPLE))
        runs = sorted({int(row["run"]) for group in training.values() for row in group["rows"]})
        self.assertIn(runs, ([1, 4], [2, 3]))
        self.assertTrue(all(len(group["rows"]) == 128 for group in training.values()))
        names = ["kappa", "log_a", "nondecision_logit"] + {"MC": ["criterion"], "MV": ["eta"],
            "MVZ": ["eta", "rho"], "MVC": ["eta", "criterion"]}[model]
        theta = [spec["parameters"][name]["population_center"] for name in names]
        theta[0] = 1. if runs == [1, 4] else 1.2
        return {"model": model, "rival": "fixed", "participants": sorted(training), "parameter_names": names,
            "parameters": [theta[:] for _ in training], "global_mean": theta[:], "version_means": [theta[:], theta[:]],
            "version_standard_deviations": [[.3] * len(names), [.3] * len(names)], "version_order": ["EI", "ER"],
            "inference": spec["method"], "converged": True, "subject_optimizer_failures": [],
            "support_contacts": [], "objective_history": [100., 100.],
            "group_interval_coverage_qualified": False, "individual_rank_recovery_qualified": False}

    def execute(self, directory, *, model="MV", index=21, epsilon=0., core_pass=True, fitter=None, monotonic=None):
        with patch.object(core, "RUNS_ROOT", directory / "runs"):
            source = core.source_directory(index)
            source.mkdir(parents=True)
            report = self.original(index, epsilon=epsilon, core_pass=core_pass)
            (source / "single_process_recovery.json").write_text(json.dumps(report))
            (source / "generating_parameters.json").write_text(json.dumps(self.truth))
            with gzip.open(source / "simulated_events.csv.gz", "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=core.shared.DESIGN_FIELDS + core.shared.RESPONSE_FIELDS)
                writer.writeheader()
                writer.writerows(row for rows in self.people.values() for row in rows)
            before = {name: (source / name).read_bytes() for name in
                      ("single_process_recovery.json", "generating_parameters.json", "simulated_events.csv.gz")}
            args = argparse.Namespace(model=model, replicate_index=index, source_dir=source,
                implementation=core.IMPLEMENTATION, output_dir=directory / "fresh", max_seconds=60)
            with patch.object(core, "require_compute"), patch.dict(core.os.environ, THREADS), \
                    patch.object(core, "fit_partially_pooled", side_effect=fitter or self.fake_fit) as fits, \
                    patch.object(core.shared, "trial_likelihood", side_effect=lambda theta, names, data, spec, rival:
                        np.full(len(data["choice"]), theta[0])), \
                    patch.object(core.time, "monotonic", side_effect=monotonic or (lambda: 0.)):
                result = core.run(args)
                with self.assertRaises(FileExistsError):
                    core.run(args)
            self.assertEqual(before, {name: (source / name).read_bytes() for name in before})
            self.assertEqual(set(p.name for p in args.output_dir.iterdir()), {core.REPORT_NAME})
            self.assertEqual(json.loads((args.output_dir / core.REPORT_NAME).read_text()), result)
        return result, fits, report

    def test_exact_array_boundary_unpadded_indices_and_seed_namespace(self):
        for index, array in ((1, 46538652), (20, 46538652), (21, 46539590), (199, 46539590)):
            with self.subTest(index=index):
                path = core.source_directory(index)
                self.assertEqual(path.name, f"replicate{index}")
                self.assertEqual(path.parent.name, f"20261005-MZ-SESOI-array-{array}")
                self.assertEqual(core.cell_seed("MZ", index), 910031 + 10007 * index)
        for index in (0, 200, True, 1., -1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                core.source_directory(index)

    def test_validator_preserves_real_index_and_refuses_relabelled_zero_or_other_seed(self):
        report = self.original(21)
        before = deepcopy(report)
        core.validate_source(report, self.truth, self.people, self.spec, 21)
        self.assertEqual(report, before)
        for kind in ("zero", "other_index", "seed", "unfinished"):
            changed = deepcopy(report)
            if kind == "zero":
                changed.update(replicate_index=0, seed=910031)
            elif kind == "other_index":
                changed.update(replicate_index=20, seed=910031 + 10007 * 20)
            elif kind == "seed":
                changed["seed"] += 1
            else:
                changed.pop("finished_utc")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.validate_source(changed, self.truth, self.people, self.spec, 21)

    def test_changed_truth_version_support_and_response_accounting_refused(self):
        for kind in ("truth", "nuisance", "truth_identity", "version", "misses", "source_scope"):
            report, truth = self.original(), deepcopy(self.truth)
            if kind == "truth":
                truth[PEOPLE[0]]["rho"] = 0.
            elif kind == "nuisance":
                truth[PEOPLE[0]]["kappa"] = self.spec["parameters"]["kappa"]["bounds"][1] + 1
            elif kind == "truth_identity":
                truth.pop(PEOPLE[0])
            elif kind == "version":
                report["participant_scores"][PEOPLE[0]]["version"] = "ER"
            elif kind == "misses":
                report["simulated_deadline_misses"] += 1
            else:
                report["observed_choices_or_RTs_used"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.validate_source(report, truth, self.people, self.spec, 21)

    def test_all_four_single_model_runs_preserve_baselines_and_have_only_descriptive_outputs(self):
        for model in core.MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                result, fits, original = self.execute(Path(temporary), model=model, core_pass=False)
            self.assertTrue(result["rival_diagnostics_complete"])
            self.assertEqual(fits.call_count, 2)
            self.assertEqual([call.args[1] for call in fits.call_args_list], [model, model])
            self.assertEqual(result["replicate_index"], 21)
            self.assertEqual(result["seed"], 910031 + 10007 * 21)
            self.assertTrue(result["saved_scores_replayed"])
            self.assertEqual(len(result["scored_primary_directions"]), 2)
            for person, row in result["participant_scores"].items():
                self.assertEqual({key: row[key] for key in ("M0", "MZ", "version")}, original["participant_scores"][person])
                self.assertAlmostEqual(row[model], 1.1)
            self.assertAlmostEqual(result["diagnostics"][model]["model_minus_saved_M0"]["mean_gain_nats_per_trial"], .8)
            self.assertEqual(result["diagnostics"][model]["model_minus_saved_MZ"]["nested_comparison"], model == "MVZ")
            for field in ("family_selection", "confusion_rate", "group_recovery", "interval_coverage", "full_gate"):
                self.assertEqual(result[field], "not_evaluated")
            self.assertFalse(result["simulation_regenerated"])
            self.assertFalse(result["saved_M0_MZ_refitted"])

    def test_replay_tolerance_preserves_recorded_last_bits_and_refuses_larger_discrepancy(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, original = self.execute(Path(temporary), index=20, epsilon=5e-13)
        self.assertTrue(result["rival_diagnostics_complete"])
        self.assertGreater(result["saved_score_replay_max_absolute_discrepancy"], 0.)
        self.assertLessEqual(result["saved_score_replay_max_absolute_discrepancy"], 1e-12)
        self.assertEqual(result["participant_scores"][PEOPLE[0]]["M0"], original["participant_scores"][PEOPLE[0]]["M0"])
        with tempfile.TemporaryDirectory() as temporary:
            result, fits, _ = self.execute(Path(temporary), epsilon=2e-12)
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertFalse(result["saved_scores_replayed"])
        self.assertEqual(fits.call_count, 0)
        self.assertIn("replay tolerance", result["error"])

    def test_nonconverged_returned_fit_is_checkpointed_and_never_scored(self):
        def nonconverged(*args):
            fit = self.fake_fit(*args)
            fit["converged"] = False
            return fit
        with tempfile.TemporaryDirectory() as temporary:
            result, fits, _ = self.execute(Path(temporary), fitter=nonconverged)
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertEqual(fits.call_count, 1)
        self.assertEqual(len(result["fits"]), 1)
        self.assertEqual(result["scored_primary_directions"], [])
        self.assertNotIn("participant_scores", result)

    def test_signal_after_first_direction_preserves_fit_without_partial_paired_scores(self):
        calls = 0
        def second_signal(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                core.interrupted(signal.SIGTERM, None)
            return self.fake_fit(*args)
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _ = self.execute(Path(temporary), fitter=second_signal)
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertEqual(len(result["fits"]), 1)
        self.assertEqual(len(result["scored_primary_directions"]), 1)
        self.assertNotIn("participant_scores", result)
        self.assertEqual(result["diagnostics"], {})

    def test_finalization_expiry_clears_completed_outputs_and_preserves_two_fits(self):
        expired = False
        def alarm(seconds):
            nonlocal expired
            if seconds == 0:
                expired = True
        with tempfile.TemporaryDirectory() as temporary, patch.object(core.signal, "alarm", side_effect=alarm):
            result, _, _ = self.execute(Path(temporary), monotonic=lambda: 61. if expired else 0.)
        self.assertFalse(result["rival_diagnostics_complete"])
        self.assertEqual(result["status"], "technical_rival_failure")
        self.assertEqual(len(result["fits"]), 2)
        self.assertEqual(len(result["scored_primary_directions"]), 2)
        self.assertEqual(result["completed_rivals"], [])
        self.assertEqual(result["diagnostics"], {})
        self.assertNotIn("participant_scores", result)
        self.assertIn("finalization", result["error"])

    def test_wrong_array_directory_is_refused_before_source_open(self):
        args = argparse.Namespace(model="MV", replicate_index=21,
            source_dir=core.RUNS_ROOT / "20261005-MZ-SESOI-array-46538652/replicate21",
            implementation=core.IMPLEMENTATION, output_dir=Path("must-not-be-created"), max_seconds=60)
        with patch.object(core, "require_compute"), self.assertRaisesRegex(ValueError, "exact selected index"):
            core.run(args)


if __name__ == "__main__":
    unittest.main()
