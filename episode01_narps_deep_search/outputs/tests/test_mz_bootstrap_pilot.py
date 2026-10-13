"""Small source-free bootstrap-pilot checks; all expensive fits are replaced."""
import argparse
from copy import deepcopy
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
import run_mz_bootstrap_pilot as core
from test_mz_group_recovery import PEOPLE, report_fixture


def fixture_halves():
    halves = []
    for runs in ((1, 4), (2, 3)):
        half = {}
        for index, person in enumerate(PEOPLE):
            version = "EI" if index < 54 else "ER"
            rows = []
            for run in runs:
                for trial in range(1, 65):
                    missing = trial == 1
                    rows.append({"participant": person, "task_version": version,
                                 "run": str(run), "trial_order": str(trial), "gain": "10", "loss": "5",
                                 "choice": "-1" if missing else "1", "no_response": "1" if missing else "0",
                                 "response_time": "" if missing else "1.25", "fixture_source": person})
            half[person] = {"version": version, "rows": rows}
        halves.append(half)
    return tuple(halves)


class MZBootstrapPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())
        cls.halves = fixture_halves()
        cls.versions = {person: group["version"] for person, group in cls.halves[0].items()}

    def fake_fit(self, training, model, rival, spec):
        self.assertEqual((model, rival), ("MZ", "fixed"))
        self.assertEqual(spec, self.spec)
        self.assertEqual(len(training), 108)
        runs = sorted({int(row["run"]) for group in training.values() for row in group["rows"]})
        self.assertIn(runs, ([1, 4], [2, 3]))
        first = runs == [1, 4]
        fit = report_fixture(self.spec)["fits"][1 if first else 3]
        fit["participants"] = sorted(training)
        for i, person in enumerate(fit["participants"]):
            group = training[person]
            self.assertEqual(len(group["rows"]), 128)
            self.assertTrue(all(row["participant"] == person for row in group["rows"]))
            self.assertEqual(len({row["fixture_source"] for row in group["rows"]}), 1)
            z = (.4 if first else .5) if group["version"] == "EI" else (.35 if first else .45)
            fit["parameters"][i][-1] = math.log(z/(1-z))
        return fit

    def execute(self, directory, *, fitter=None, effect=None, source_change=None, monotonic=None):
        source = directory / "source"
        source.mkdir()
        path = source / "single_process_recovery.json"
        baseline = report_fixture(self.spec)
        if source_change:
            source_change(baseline)
        path.write_text(json.dumps(baseline))
        before = path.read_bytes()
        args = argparse.Namespace(source_report=path, implementation=core.IMPLEMENTATION,
                                  output_dir=directory / "fresh", max_seconds=60)
        with patch.object(core, "SOURCE_REPORT", path), patch.object(core, "require_compute"), \
                patch.object(core, "load_split", return_value=self.halves) as loader, \
                patch.object(core, "fit_partially_pooled", side_effect=fitter or self.fake_fit) as fit_calls, \
                patch.object(core, "physical_group_effects", side_effect=effect or core.physical_group_effects), \
                patch.object(core.time, "monotonic", side_effect=monotonic or [0.] * 20):
            result = core.run(args)
            with self.assertRaises(FileExistsError):
                core.run(args)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(set(p.name for p in args.output_dir.iterdir()), {core.REPORT_NAME})
        saved = json.loads((args.output_dir / core.REPORT_NAME).read_text())
        self.assertEqual(saved, result)
        return result, fit_calls, loader

    def test_exact_seed_strata_replacement_and_distinct_draw_slot_ids(self):
        draws = core.paired_draw(self.versions)
        generator = np.random.default_rng(np.random.SeedSequence([1200037, 910031, 0]))
        expected = []
        for version in ("EI", "ER"):
            roster = sorted(p for p, value in self.versions.items() if value == version)
            expected.extend(generator.choice(roster, size=54, replace=True).tolist())
        self.assertEqual([draw["source_participant"] for draw in draws], expected)
        self.assertEqual(len({draw["bootstrap_participant"] for draw in draws}), 108)
        self.assertLess(len(set(expected)), 108)
        self.assertEqual(draws, core.paired_draw(dict(reversed(list(self.versions.items())))))
        self.assertEqual([draw["stratum_draw_index"] for draw in draws], list(range(54)) * 2)

    def test_same_whole_person_draw_both_halves_and_original_rows_unchanged(self):
        draws = core.paired_draw(self.versions)
        core.validate_source_halves(self.halves, self.versions)
        first, second = [core.resampled_half(half, draws) for half in self.halves]
        self.assertEqual(set(first), set(second))
        for draw in draws:
            pseudo, source = draw["bootstrap_participant"], draw["source_participant"]
            rows = first[pseudo]["rows"] + second[pseudo]["rows"]
            self.assertEqual(len(rows), 256)
            self.assertEqual({row["fixture_source"] for row in rows}, {source})
            self.assertEqual({(int(row["run"]), int(row["trial_order"])) for row in rows},
                             {(run, trial) for run in range(1, 5) for trial in range(1, 65)})
            self.assertEqual(sum(row["choice"] == "-1" for row in rows), 4)
            self.assertTrue(all(row["participant"] == source for half in self.halves for row in half[source]["rows"]))
        duplicates = {}
        for draw in draws:
            duplicates.setdefault(draw["source_participant"], []).append(draw["bootstrap_participant"])
        pair = next(identities[:2] for identities in duplicates.values() if len(identities) > 1)
        self.assertIsNot(first[pair[0]]["rows"], first[pair[1]]["rows"])
        self.assertIsNot(first[pair[0]]["rows"][0], first[pair[1]]["rows"][0])

    def test_complete_pilot_refits_only_two_mz_halves_and_has_no_interval_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, fits, loader = self.execute(Path(temporary))
        self.assertTrue(report["bootstrap_draw_complete"])
        self.assertEqual(report["status"], "paired_refit_draw_completed")
        self.assertEqual(fits.call_count, 2)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[0].name, "simulated_events.csv.gz")
        self.assertEqual(loader.call_args.args[1:], ([1, 4], [2, 3]))
        self.assertEqual([fit["training_runs"] for fit in report["fits"]], [[1, 4], [2, 3]])
        self.assertEqual(set(fits.call_args_list[0].args[0]), set(fits.call_args_list[1].args[0]))
        self.assertAlmostEqual(report["physical_rejection_displacement"]["EI"], .05)
        self.assertAlmostEqual(report["physical_rejection_displacement"]["ER"], .10)
        self.assertFalse(report["interval_coverage_evaluated"])
        self.assertFalse(report["group_interval_coverage_qualified"])
        self.assertIsNone(report["interval_bootstrap_count_selected"])
        self.assertIsNone(report["interval_convention_selected"])
        self.assertFalse(report["heldout_scoring_performed"])
        self.assertEqual(report["full_recovery_gate"], "not_evaluated")
        self.assertLess(len(json.dumps(report).encode()), core.OUTPUT_LIMIT_BYTES)

    def test_nonconvergence_preserves_returned_fit_as_technical_not_success(self):
        def nonconverged(*args):
            fit = self.fake_fit(*args)
            fit["converged"] = False
            return fit
        with tempfile.TemporaryDirectory() as temporary:
            report, fits, _ = self.execute(Path(temporary), fitter=nonconverged)
        self.assertFalse(report["bootstrap_draw_complete"])
        self.assertEqual(report["status"], "technical_bootstrap_pilot_failure")
        self.assertEqual(len(report["fits"]), 1)
        self.assertEqual(fits.call_count, 1)
        self.assertNotIn("physical_rejection_displacement", report)

    def test_signal_after_first_fit_and_late_failure_keep_completed_checkpoints_only(self):
        count = 0
        def terminate_second(*args):
            nonlocal count
            count += 1
            if count == 2:
                core.interrupted(signal.SIGTERM, None)
            return self.fake_fit(*args)
        with tempfile.TemporaryDirectory() as temporary:
            report, _, _ = self.execute(Path(temporary), fitter=terminate_second)
        self.assertFalse(report["bootstrap_draw_complete"])
        self.assertEqual(len(report["fits"]), 1)
        with tempfile.TemporaryDirectory() as temporary:
            report, _, _ = self.execute(Path(temporary), effect=RuntimeError("late deadline"))
        self.assertFalse(report["bootstrap_draw_complete"])
        self.assertEqual(len(report["fits"]), 2)
        self.assertNotIn("physical_rejection_displacement", report)

    def test_deadline_and_unfinished_source_refuse_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, fits, loader = self.execute(Path(temporary), monotonic=[0., 61.])
        self.assertEqual(fits.call_count, 0)
        self.assertEqual(loader.call_count, 0)
        self.assertFalse(report["bootstrap_draw_complete"])
        with tempfile.TemporaryDirectory() as temporary:
            report, fits, loader = self.execute(Path(temporary), source_change=lambda r: r.pop("finished_utc"))
        self.assertEqual(fits.call_count, 0)
        self.assertEqual(loader.call_count, 0)
        self.assertFalse(report["bootstrap_draw_complete"])

    def test_expiry_when_alarm_is_cancelled_clears_final_completion_and_effect(self):
        expired = False
        def alarm(seconds):
            nonlocal expired
            if seconds == 0:
                expired = True
        with tempfile.TemporaryDirectory() as temporary, patch.object(core.signal, "alarm", side_effect=alarm):
            report, _, _ = self.execute(Path(temporary), monotonic=lambda: 61. if expired else 0.)
        self.assertEqual(report["status"], "technical_bootstrap_pilot_failure")
        self.assertFalse(report["bootstrap_draw_complete"])
        self.assertEqual(len(report["fits"]), 2)
        self.assertNotIn("physical_rejection_displacement", report)
        self.assertIn("finalization", report["error"])

    def test_source_identity_censoring_and_run_slots_are_not_dropped_or_repaired(self):
        for kind in ("duplicate", "version", "missing", "miss_censoring", "response_time", "offer"):
            halves = list(self.halves)
            first = dict(halves[0])
            first[PEOPLE[0]] = deepcopy(first[PEOPLE[0]])
            halves[0] = first
            group = first[PEOPLE[0]]
            if kind == "duplicate":
                group["rows"][1] = dict(group["rows"][0])
            elif kind == "version":
                group["version"] = "ER"
            elif kind == "missing":
                group["rows"].pop()
            elif kind == "miss_censoring":
                group["rows"][0]["response_time"] = "1"
            elif kind == "response_time":
                group["rows"][1]["response_time"] = "nan"
            else:
                group["rows"][1]["gain"] = "-1"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.validate_source_halves(halves, self.versions)

    def test_no_other_source_or_extra_draw_index_interface(self):
        args = argparse.Namespace(source_report=Path("unselected-source.json"), implementation=core.IMPLEMENTATION,
                                  output_dir=Path("must-not-exist"), max_seconds=60)
        with patch.object(core, "require_compute"), self.assertRaisesRegex(ValueError, "Only the original"):
            core.run(args)
        versions = dict(self.versions)
        versions[PEOPLE[0]] = "ER"
        with self.assertRaises(ValueError):
            core.paired_draw(versions)


if __name__ == "__main__":
    unittest.main()
