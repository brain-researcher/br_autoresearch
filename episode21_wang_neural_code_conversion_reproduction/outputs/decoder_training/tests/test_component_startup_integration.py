"""Five work48 startup-replacement integration checks; Slurm execution only.

All journals, reviews, closed records and scheduler identities are fixtures.
No real journal, payload, process, scheduler or scientific artifact is opened.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_parallelism_amendment as parallelism
import reviewed_component_startup_replacement as startup
import reviewed_io_quarantine as quarantine
import submit_full_decoder_jobs as submit
import test_reviewed_io_quarantine as incident_fixtures


class ComponentStartupIntegrationTests(unittest.TestCase):
    def setUp(self):
        incident = incident_fixtures.ReviewedIOQuarantineTests()
        incident.setUp()
        self.authority, self.manifest = copy.deepcopy((incident.authority, incident.manifest))
        self.rows = copy.deepcopy(incident.rows)
        self.base, self.fixture = incident.base, incident.fixture
        self.partial_review, self.quarantine_review = copy.deepcopy((incident.source.review2, incident.review))
        # Only fabricated non-incident assignments get minimal channel metadata
        # for real parallelism accounting. Existing closed-fragment fixtures stay
        # unchanged. Rebind the artificial immutable basis before replaying it.
        self.pressure_works = [f"work_{number:04d}" for number in (*range(63, 76), 20, 21)]
        for work in ["work_0009", "work_0010", "work_0048", "work_0049", *self.pressure_works]:
            item = self.manifest["work_items"][int(work[-4:]) - 1]
            item["spans"] = [{"participant": item["participant"],
                "layer": "conv3_1" if work in ("work_0009", "work_0010") else "conv1_1",
                "channel": 0, "spatial_Fortran_interval": [0, 1]}]
        self.rows[1]["authority"] = copy.deepcopy(self.authority)
        self.rows[1]["manifest"] = copy.deepcopy(self.manifest)
        self.fixture.authority, self.fixture.manifest = self.authority, self.manifest
        self.fixture.documents = copy.deepcopy(self.fixture.documents)
        self.rows.append(parallelism.make_event(observed_utc="2026-10-08T01:12:00Z",
            user_instruction="Fixture: parallel work within unchanged aggregate limits"))
        previous = self.ordinary()
        submit.select_component_wrapper(previous, parallelism.FIXED_POLICY)
        terminal = self.base.terminal(startup.JOB, elapsed=39, state="FAILED", allocation_job_id="47080265")
        terminal["exit_code"] = "74:0"
        self.rows.extend([self.event(previous, 8), self.base.accepted("47080265", "batch_0008"), terminal])
        self.account = self.replay(self.rows)
        self.review = {**copy.deepcopy(startup.FIXED), "reviewed_utc": "2026-10-09T06:30:00Z",
            "fix_evidence": "Fixture: portable tee with checked PIPESTATUS",
            "terminal_evidence": "Fixture: FAILED74:0 on one CPU for39 seconds",
            "absence_evidence": "Fixture: no shard, request, status or fragment",
            "user_authorization_context": "Fixture: one manual repair under the existing grant"}

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    @staticmethod
    def event(batch, number=9):
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", **copy.deepcopy(batch)}

    def ordinary(self, selected=(startup.WORK,)):
        batch = submit.build_batch(self.authority, self.manifest, "workers", selected, "normal",
            manual_refill=True, deadline_partial_review=copy.deepcopy(self.partial_review),
            effective_parallelism=parallelism.FIXED_POLICY, constraint="CPU_GEN:SIE")
        batch[quarantine.FIELD] = copy.deepcopy(self.quarantine_review)
        return batch

    def replacement(self):
        batch = self.ordinary()
        submit.select_component_replacement_wrapper(batch, copy.deepcopy(self.review), parallelism.FIXED_POLICY)
        return batch

    def admit(self, batch, account=None, results=None):
        with self.fixture.metadata():
            return submit.admit_batch(self.account if account is None else account, self.manifest, batch,
                results=self.fixture.results if results is None else results)

    def test_only_acceptance_supersedes_and_every_state_preserves_exact_charge(self):
        before = copy.deepcopy((self.rows, self.account, self.authority, self.manifest))
        self.assertEqual(self.account["actual_CPU_seconds"] - self.replay(self.rows[:-1])["actual_CPU_seconds"], 39)
        intent_rows = self.rows + [self.event(self.replacement())]
        pending = self.replay(intent_rows)
        self.assertEqual(pending["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual(pending["outstanding_reserved_CPU_seconds"],
                         self.account["outstanding_reserved_CPU_seconds"] + 28800)
        self.assertEqual(pending["assigned"], self.account["assigned"])
        self.assertNotIn(startup.JOB, pending["superseded_pre_admission_jobs"])
        self.assertEqual(len(pending["uncertain_batches"]), 1)
        accepted = self.replay(intent_rows + [self.base.accepted("970", "batch_0009")])
        self.assertEqual(accepted["superseded_pre_admission_jobs"],
                         self.account["superseded_pre_admission_jobs"] | {startup.JOB})
        self.assertEqual(set(accepted["jobs"]) - set(self.account["jobs"]), {"970_47"})
        self.assertEqual(accepted["outstanding_reserved_CPU_seconds"], pending["outstanding_reserved_CPU_seconds"])
        for account in (pending, accepted):
            self.assertEqual(account["jobs"][startup.JOB], self.account["jobs"][startup.JOB])
            self.assertEqual(account["additional_remainder_durable_bytes"], self.account["additional_remainder_durable_bytes"])
            self.assertEqual(account["additional_io_recovery_durable_bytes"], 0)
            self.assertEqual(account["basis"], self.account["basis"])
            self.assertEqual(account["budget"], self.account["budget"])
            self.assertEqual(account["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        rejected = self.replay(intent_rows + [{"event": "submission_not_accepted",
            "batch_id": "batch_0009", "evidence": "Fixture: scheduler explicitly rejected request"}])
        self.assertNotIn(startup.JOB, rejected["superseded_pre_admission_jobs"])
        self.assertEqual(rejected["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual((self.rows, self.account, self.authority, self.manifest), before)

    def test_duplicate_without_review_wrong_charge_and_chained_attempt_still_refuse(self):
        absent = self.replacement()
        del absent[startup.FIELD]
        with self.assertRaisesRegex(quarantine.QuarantineError, "no_retry_or_reassignment"):
            self.admit(absent)
        with self.assertRaisesRegex(quarantine.QuarantineError, "no_retry_or_reassignment"):
            self.replay(self.rows + [self.event(absent)])
        wrong_charge = copy.deepcopy(self.account)
        wrong_charge["jobs"][startup.JOB]["terminal"]["elapsed_seconds"] = 38
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "39_second_failure"):
            self.admit(self.replacement(), wrong_charge)
        accepted_rows = self.rows + [self.event(self.replacement()), self.base.accepted("970", "batch_0009")]
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "only_no_chaining"):
            self.replay(accepted_rows + [self.event(self.replacement(), 10)])

    def test_readiness_bypasses_only_known_predecessor_and_keeps_other_closeouts(self):
        observed = []
        def results(account, manifest, **kwargs):
            observed.append(set(account["jobs"]))
            return self.fixture.results(account, manifest, **kwargs)
        self.admit(self.replacement(), results=results)
        closed = {key for key, job in self.account["jobs"].items() if job["terminal"] is not None}
        self.assertEqual(observed, [closed - {startup.JOB}])
        self.assertTrue({entry[0] for entry in quarantine.KNOWN.values()} <= observed[0])
        self.assertIn("46839670_4", observed[0])
        # Once the replacement is accepted, its still-running job is deferred;
        # the old failed startup remains charged but no longer blocks fresh work.
        accepted_rows = self.rows + [self.event(self.replacement()), self.base.accepted("970", "batch_0009")]
        accepted = self.replay(accepted_rows)
        self.admit(self.ordinary(("work_0049",)), accepted)
        failed_rows = accepted_rows + [self.base.terminal("800_8", elapsed=17,
            state="FAILED", allocation_job_id="880")]
        with self.assertRaisesRegex(submit.Stop, "failed_parent_requires_review"):
            self.admit(self.ordinary(("work_0049",)), self.replay(failed_rows))

    def test_cpu_and_aggregate_scratch_bounds_are_not_waived(self):
        tight = copy.deepcopy(self.account)
        tight["budget"]["cpu_seconds"] = tight["actual_CPU_seconds"] + tight["outstanding_reserved_CPU_seconds"] + 28799
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            self.admit(self.replacement(), tight)
        pressure = copy.deepcopy(self.account)
        pressure["batches"]["fixture_pressure"] = {"phase": "workers", "accepted_id": "980", "not_submitted": False}
        for work in self.pressure_works:
            index = int(work[-4:]) - 1
            key = f"980_{index}"
            pressure["jobs"][key] = {"job_id": key, "logical_id": work, "phase": "workers",
                "batch_id": "fixture_pressure", "reserved_CPU_seconds": 28800, "terminal": None}
            pressure["assigned"].add(work)
            pressure["outstanding_reserved_CPU_seconds"] += 28800
        self.assertLess(len([job for job in pressure["jobs"].values() if job["terminal"] is None]) + 1, 42)
        with self.assertRaisesRegex(parallelism.ParallelismError, "scratch_exceeds_16GB"):
            self.admit(self.replacement(), pressure)

    def test_wrapper003_selection_is_only_exact_work48_with_fixed_resources(self):
        ordinary, replacement = self.ordinary(), self.replacement()
        self.assertEqual(replacement["command"][:-1], ordinary["command"][:-1])
        self.assertEqual(replacement["command"][-1], str(submit.COMPONENT_REPAIR_WRAPPER))
        self.assertEqual(replacement["allocations"], [startup.ALLOCATION])
        for option in ("--cpus-per-task=1", "--mem=7G", "--time=480", "--array=47%1", "--no-requeue"):
            self.assertIn(option, replacement["command"])
        self.assertEqual(replacement[quarantine.FIELD], self.quarantine_review)
        self.assertEqual(replacement[startup.PARTIAL_FIELD], self.partial_review)
        for selected in (("work_0049",), (startup.WORK, "work_0049")):
            with self.subTest(selected=selected), self.assertRaisesRegex(submit.Stop, "only_in_fresh_amended_work48"):
                submit.select_component_replacement_wrapper(self.ordinary(selected), self.review, parallelism.FIXED_POLICY)
        wrong_wrapper = self.replacement()
        wrong_wrapper["command"][-1] = str(submit.COMPONENT_WRAPPER)
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "wrapper003"):
            self.admit(wrong_wrapper)


if __name__ == "__main__":
    unittest.main()
