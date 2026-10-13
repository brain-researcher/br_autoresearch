"""Focused omission/refit orchestration checks; run only in coordinated compute."""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz0_offer_sequence_refit as refit
import test_mz_other_partitions as previous_tests
from test_mz0_offer_sequence_influence import fixture as influence_fixture


class OfferSequenceRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_tests.OtherPartitionsTests.setUpClass()
        cls.prior = previous_tests.OtherPartitionsTests("test_both_partitions_replay_primary_then_fit_four_fresh_original500_models")
        cls.spec, cls.people = cls.prior.spec, cls.prior.people

    def inputs(self):
        source, truth = self.prior.source()
        _, schedule, _ = influence_fixture()
        schedule["groups"], members = [], []
        for version in ("EI", "ER"):
            people = sorted(person for person, rows in self.people.items() if rows[0]["task_version"] == version)
            offset = 0
            for index, count in enumerate((5, 6, 6, 7, 7, 7, 8, 8), 1):
                group = f"{version}-offers-{index:02d}"
                selected = people[offset:offset + count]
                offset += count
                schedule["groups"].append({"offer_sequence_id": group, "participants": selected})
                members.extend({"participant": person, "task_version": version, "offer_sequence_id": group}
                               for person in selected)
        return source, truth, schedule, members

    def args(self, output, group="EI-offers-01", **changes):
        values = dict(omitted_group=group, source_dir=refit.SOURCE, implementation=refit.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def retained_fit(self, retained, model="M0", train=(1, 4), held=(2, 3), converged=True):
        fit = self.prior.fixture.fit(model, train, held, converged)
        selected = [index for index, person in enumerate(fit["participants"]) if person in retained]
        fit["participants"] = [fit["participants"][index] for index in selected]
        fit["parameters"] = [fit["parameters"][index] for index in selected]
        return fit

    def exercise(self, output, group="EI-offers-01", *, bad_source=False, bad_membership=False,
                 replay_error=False, nonconverged_at=None, score_error_at=None, worker_error_at=None,
                 deadline_stage=None, checkpoint_delay=0.):
        source, truth, schedule, members = self.inputs()
        if bad_source:
            source["seed"] += 1
        if bad_membership:
            members.append(deepcopy(members[0]))
        preserved = deepcopy((source, truth, schedule, members, self.people, self.spec))
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        fits, scored, checkpoints = [], [], []
        now, delayed = [100.], [False]
        real_read, real_writer, real_scorer, real_alarm = Path.read_text, refit.write_report, refit.shared.score_fit, signal.alarm
        expected_removed = next(row["participants"] for row in schedule["groups"] if row["offer_sequence_id"] == group)
        retained = {person: rows for person, rows in self.people.items() if person not in expected_removed}

        def reader(path, *args, **kwargs):
            if path == refit.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == refit.SOURCE / refit.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 5401.
                return json.dumps(source)
            if path == refit.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            return real_read(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, refit.SOURCE / "simulated_events.csv.gz")
            return tuple({person: {"version": rows[0]["task_version"],
                "rows": [row for row in rows if int(row["run"]) in runs]}
                for person, rows in self.people.items()} for runs in (train, held))

        def fitter(training, model, rival, spec, *, workers, max_seconds, progress):
            self.assertEqual((rival, spec, workers), ("fixed", self.spec, 4))
            self.assertEqual(set(training), set(retained))
            self.assertEqual(len(scored), 4 + len(fits))
            self.assertTrue(all(len(item["rows"]) == 128 for item in training.values()))
            train = tuple(sorted({int(row["run"]) for item in training.values() for row in item["rows"]}))
            held = tuple(run for run in (1, 2, 3, 4) if run not in train)
            self.assertAlmostEqual(max_seconds, 5400. - (now[0] - 100.) - refit.parallel.CLEANUP_SECONDS - 1.)
            fits.append((model, train, held, sorted(training), max_seconds))
            if worker_error_at == len(fits):
                error = RuntimeError("synthetic bounded worker failure")
                error.parallel_progress = {"stage": "conditional", "sweep": 1, "workers_cleaned_up": True,
                    "completed_results": {}, "pending_participants": sorted(training), "objective_history": []}
                raise error
            progress({"stage": "sweep", "sweep": 1, "objective": 100., "converged": False,
                      "subject_optimizer_failures": []})
            return self.retained_fit(retained, model, train, held, converged=nonconverged_at != len(fits))

        def scorer(fit, scoring, spec, check):
            self.assertEqual(spec, self.spec)
            primary = not fits
            self.assertEqual(set(scoring), set(self.people if primary else retained))
            self.assertTrue(all(len(item["rows"]) == 128 for item in scoring.values()))
            scored.append((fit["model"], tuple(fit["training_runs"]), primary))
            if not primary and score_error_at == len(fits):
                raise RuntimeError("synthetic scoring interruption")
            def likelihood(theta, names, data, impl, rival):
                self.assertEqual((impl, rival), (self.spec, "fixed"))
                if primary:
                    value = (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.)
                else:
                    value = -2.4 + ((.04 if data["gain"].max() > 20 else .08) if "rho" in names else 0.)
                    if deadline_stage == "scoring":
                        now[0] += 5401.
                return np.full(128, value)
            with patch.object(refit.shared, "trial_likelihood", side_effect=likelihood):
                return real_scorer(fit, scoring, spec, check)

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            real_writer(directory, report)
            if report["active_fit"] is not None and not report["fresh_fit_calls"] and not delayed[0]:
                now[0] += 5401. if deadline_stage == "prefit" else checkpoint_delay
                delayed[0] = True
            if deadline_stage == "finalization" and report["offer_sequence_refit_complete"]:
                now[0] += 5401.

        def alarm(seconds):
            result = real_alarm(seconds)
            if seconds == 0 and deadline_stage == "cleanup":
                now[0] += 5401.
            return result

        with ExitStack() as stack:
            for context in (patch.object(refit, "require_compute"),
                    patch.dict(os.environ, {**{key: "1" for key in refit.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}),
                    patch.object(refit.shared, "EPISODE", output.parent.parent.parent),
                    patch.object(Path, "read_text", reader),
                    patch.object(refit.shared, "load_cohort", return_value=self.people),
                    patch.object(refit.shared, "load_split", side_effect=split),
                    patch.object(refit, "read_membership", return_value=(schedule, members)),
                    patch.object(refit.shared, "score_fit", side_effect=scorer),
                    patch.object(refit.parallel, "fit_partially_pooled_parallel", side_effect=fitter),
                    patch.object(refit, "write_report", side_effect=writer),
                    patch.object(refit.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(refit.signal, "alarm", side_effect=alarm)):
                stack.enter_context(context)
            report = refit.run(self.args(output, group))
        self.assertEqual((source, truth, schedule, members, self.people, self.spec), preserved)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / refit.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assertTrue(all(record[field] == "not_evaluated" for field in refit.CLAIMS))
            self.assertFalse(any(record[field] for field in ("simulation_regenerated", "saved_fit_parameters_reused",
                "observed_choices_or_RTs_used", "automatic_retry", "automatic_ceiling_increase", "subsequent_omissions_selected")))
        return report, fits, scored, checkpoints

    def test_all16_complements_preserve_version_identity_and_every_retained_trial(self):
        source, _, schedule, members = self.inputs()
        for group in refit.influence.GROUP_IDS:
            retained, omitted, counts = refit.retained_cohort(source, schedule, members, self.people, group)
            expected = next(row["participants"] for row in schedule["groups"] if row["offer_sequence_id"] == group)
            self.assertEqual(omitted, expected)
            self.assertEqual(set(retained), set(self.people) - set(expected))
            self.assertEqual(counts[group[:2]], 54 - len(expected))
            self.assertEqual(counts["ER" if group[:2] == "EI" else "EI"], 54)
            self.assertTrue(all(retained[p] is self.people[p] and len(retained[p]) == 256 for p in retained))

    def test_four_fresh_original_fits_and_unequal_version_weights_use_actual_retained_counts(self):
        for group, expected_counts in (("EI-offers-01", {"EI": 49, "ER": 54}),
                                       ("ER-offers-08", {"EI": 54, "ER": 46})):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, scored, checkpoints = self.exercise(Path(directory) / "outputs/runs/fresh", group, checkpoint_delay=7.)
            self.assertTrue(report["offer_sequence_refit_complete"])
            self.assertEqual(report["participants_by_version"], expected_counts)
            self.assertEqual([item[:3] for item in fits], [(model, tuple(train), tuple(held))
                for train, held in refit.shared.HALVES for model in ("M0", "MZ")])
            self.assertEqual((len(report["fits"]), len(report["completed_components"]), len(scored)), (4, 4, 8))
            self.assertEqual(report["implementation"], self.spec)
            gain = sum(expected_counts[v] * effect for v, effect in (("EI", .04), ("ER", .08))) / sum(expected_counts.values())
            self.assertAlmostEqual(report["nested_diagnostic"]["mean_gain_nats_per_trial"], gain, places=12)
            for row in report["participant_scores"].values():
                self.assertAlmostEqual(row["M0"], -2.4)
                self.assertAlmostEqual(row["MZ"] - row["M0"], .04 if row["version"] == "EI" else .08)
            self.assertTrue(any(len(record["fits"]) > len(record["completed_components"]) for record in checkpoints))

    def test_bad_source_replay_and_membership_never_enter_fresh_fitter(self):
        for option in ({"bad_source": True}, {"bad_membership": True}, {"replay_error": True}):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "outputs/runs/fresh", **option)
            self.assertFalse(report["offer_sequence_refit_complete"])
            self.assertEqual(fits, [])
            self.assertNotIn("participant_scores", report)

    def test_local_fit_validator_preserves_identity_support_and_original_ceiling(self):
        source, _, schedule, members = self.inputs()
        retained, *_ = refit.retained_cohort(source, schedule, members, self.people, "EI-offers-01")
        fit = self.retained_fit(retained)
        refit.validate_retained_fit(fit, "M0", (1, 4), (2, 3), self.spec, retained)
        for change in (lambda f: f["participants"].reverse(), lambda f: f["parameters"].pop(),
                       lambda f: f.update(converged=False), lambda f: f.update(subject_optimizer_failures=[{}]),
                       lambda f: f.update(objective_history=[100.] * 501),
                       lambda f: f["parameters"][0].__setitem__(0, 1e9)):
            changed = deepcopy(fit)
            change(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                refit.validate_retained_fit(changed, "M0", (1, 4), (2, 3), self.spec, retained)

    def test_later_technical_failures_keep_returned_fits_and_only_completed_scores(self):
        for option, fit_count in (({"nonconverged_at": 2}, 2), ({"score_error_at": 2}, 2), ({"worker_error_at": 2}, 1)):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "outputs/runs/fresh", **option)
            self.assertEqual((len(fits), len(report["fits"]), len(report["completed_components"])), (2, fit_count, 1))
            self.assertFalse(report["offer_sequence_refit_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)
            if "worker_error_at" in option:
                self.assertIs(report["parallel_workers_cleaned_up"], True)
                self.assertEqual(report["parallel_failure"]["stage"], "conditional")

    def test_whole_deadlines_cover_reads_prefit_scoring_publication_and_cleanup(self):
        for stage in ("source", "prefit", "scoring", "finalization", "cleanup"):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "outputs/runs/fresh", deadline_stage=stage)
            self.assertFalse(report["offer_sequence_refit_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)
            if stage in ("source", "prefit"):
                self.assertEqual(fits, [])
            if stage in ("finalization", "cleanup"):
                self.assertEqual(len(report["completed_components"]), 4)

    def test_cpu_cap_group_source_and_existing_output_guards(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(refit, "require_compute"), \
                patch.dict(os.environ, {**{key: "1" for key in refit.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}):
            output = Path(directory) / "outputs/runs/fresh"
            for change in ({"max_seconds": 5401}, {"max_seconds": 0}, {"omitted_group": "EI-offers-09"},
                           {"source_dir": Path(directory)}, {"implementation": Path(directory)}):
                with self.assertRaises(ValueError):
                    refit.run(self.args(output, **change))
            with patch.dict(os.environ, SLURM_CPUS_PER_TASK="1"), self.assertRaises(ValueError):
                refit.run(self.args(output))
            with patch.dict(os.environ, OMP_NUM_THREADS="4"), self.assertRaises(ValueError):
                refit.run(self.args(output))
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                refit.run(self.args(output))


if __name__ == "__main__":
    unittest.main()
