"""Integrated finite remainder admission fixtures; run only in coordinated Slurm.

All job histories, scheduler replies and closed records are artificial. No real
journal, authority, status, payload or scheduler is opened by these fixtures.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainder
import reviewed_deadline_partials as partials
import submit_full_decoder_jobs as submit
import test_decoder_deadline_remainders as remainder_fixtures
import test_submit_full_decoder_jobs as base_fixtures


class SubmitDeadlineRemainderTests(unittest.TestCase):
    def setUp(self):
        self.base = base_fixtures.FullDecoderSubmissionTests()
        self.authority, self.manifest, self.plan, self.review, predecessors, _ = remainder_fixtures.fixture()
        self.rows = self.base.rows(self.authority, self.manifest) + [
            self.base.intent(self.authority, self.manifest), self.base.accepted(), self.base.terminal(),
            self.base.intent(self.authority, self.manifest, "workers",
                             [f"work_{i+1:04d}" for i in range(8)], number=2),
            self.base.accepted("46839670", "batch_0002")]
        for index in range(8):
            work = f"work_{index+1:04d}"
            if work in remainder.WORK_IDS:
                _, job, raw, _ = remainder.KNOWN[work]
                self.rows.append(self.base.terminal(job, elapsed=28505 if index == 4 else 28503,
                                                    state="FAILED", allocation_job_id=raw))
            else:
                self.rows.append(self.base.terminal(f"46839670_{index}", allocation_job_id=str(1000+index)))
        reviewed = submit.build_batch(self.authority, self.manifest, "workers", ["work_0009"], "normal",
                                      manual_refill=True, deadline_partial_review=self.review)
        self.rows += [{"event": "submission_intent", "batch_id": "batch_0003", **reviewed},
                      self.base.accepted("800", "batch_0003")]
        self.account = self.replay(self.rows)
        self.documents = {submit.DEADLINE_PARTIAL_REVIEW: self.review}
        for item in self.plan["work_items"]:
            self.documents[submit.BASE / item["predecessor_status"]] = predecessors[item["work_item_id"]]
        self.real_load_context = submit.load_remainder_context

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def batch(self, selected=("work_0005",)):
        return submit.build_deadline_remainders(self.authority, self.manifest, self.plan, selected,
                                                "normal", constraint="CPU_GEN:SIE")

    def event(self, batch, number=4):
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", **batch}

    def reader(self, path, account=None):
        if path in self.documents:
            return copy.deepcopy(self.documents[path])
        work = "states" if path.name == "state_parent_status.json" else path.parent.name
        job = next(job for job in account["jobs"].values() if job["logical_id"] == work)
        record = {"allocation": {"job_id": job["terminal"]["allocation_job_id"]}}
        if work == "states":
            record.update(status="complete_four_participant_training_states_only",
                          participants_completed=list(submit.state.NEW_PARTICIPANTS), participants_remaining=[])
        else:
            item = next(item for item in self.manifest["work_items"] if item["work_item_id"] == work)
            record.update(status="complete_work_item_only", work_item_id=work, participant=item["participant"],
                resources=item["resources"], requested_spans=item["spans"], committed_spans=item["spans"],
                units_requested=item["units_requested"], completed_units=item["units_requested"],
                remaining_spans=[], completed_units_is_lower_bound=False, closeout_errors=[])
        return record

    @contextmanager
    def metadata(self):
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "is_symlink", return_value=False), \
                mock.patch.object(submit, "load_remainder_context", side_effect=lambda account, manifest, plan:
                    self.real_load_context(account, manifest, plan, reader=self.reader)):
            yield

    def results(self, account, manifest, **kwargs):
        return submit.require_previous_results(account, manifest,
            reader=lambda path: self.reader(path, account), **kwargs)

    def close_remainder_record(self, work, raw):
        item = next(row for row in self.plan["work_items"] if row["work_item_id"] == work)
        record = copy.deepcopy(self.documents[submit.BASE / item["predecessor_status"]])
        record.update(status="complete_work_item_only", resources=item["resources"],
            requested_spans=item["spans"], committed_spans=item["spans"], remaining_spans=[],
            units_requested=item["units_requested"], completed_units=item["units_requested"],
            remainder_plan=str(submit.REMAINDER_PLAN), predecessor_job_id=item["predecessor_job_id"],
            predecessor_status=item["predecessor_status"])
        record.pop("error")
        record.pop("error_type")
        record["allocation"]["job_id"] = raw
        self.documents[submit.BASE / item["output_directory"] / "work_item_status.json"] = record
        return record

    def test_exact_subset_preserves_assignments_charges_and_reserves_all_extra_storage_once(self):
        before = copy.deepcopy((self.authority, self.manifest, self.account))
        batch = self.batch(("work_0005", "work_0007"))
        self.assertEqual(batch["command"][-1], str(submit.REMAINDER_WRAPPER))
        self.assertIn("--array=4,6%2", batch["command"])
        self.assertIn("--constraint=CPU_GEN:SIE", batch["command"])
        with self.metadata():
            submit.admit_batch(self.account, self.manifest, batch, results=self.results)
        updated = self.replay(self.rows + [self.event(batch), self.base.accepted("900", "batch_0004")])
        self.assertEqual(updated["assigned"], self.account["assigned"])
        self.assertEqual(updated["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"], 3*28800)
        self.assertEqual(updated["additional_remainder_durable_bytes"], 2368108032)
        self.assertEqual(submit.summary(updated)["accepted_remainder_attempts"], 2)
        for work in remainder.WORK_IDS:
            job = remainder.KNOWN[work][1]
            self.assertEqual(updated["jobs"][job], self.account["jobs"][job])
            self.assertNotIn(job, updated["superseded_pre_admission_jobs"])
        self.assertEqual((self.authority, self.manifest, self.account), before)

    def test_replay_refuses_default_retry_wrong_modes_repeated_attempt_and_plan_changes(self):
        batch = self.batch()
        cases = []
        for field in (remainder.FIELD, accounting.MANUAL_REFILL_FIELD):
            altered = copy.deepcopy(batch)
            del altered[field]
            cases.append(altered)
        cases.append({**batch, accounting.STARTUP_REPLACEMENT_FIELD: {}})
        altered = copy.deepcopy(batch)
        altered["allocations"][0]["reserved_CPU_seconds"] = 3600
        cases.append(altered)
        for altered in cases:
            with self.subTest(fields=tuple(altered)), self.assertRaises((accounting.AccountingError, partials.ReviewError)):
                self.replay(self.rows + [self.event(altered)])
        history = self.rows + [self.event(batch), self.base.accepted("900", "batch_0004"),
                              self.base.terminal("900_4", allocation_job_id="901")]
        with self.assertRaisesRegex(partials.ReviewError, "one_original_closed_attempt"):
            self.replay(history + [self.event(batch, 5)])
        altered = self.batch(("work_0006",))
        altered[remainder.FIELD] = copy.deepcopy(self.plan)
        altered[remainder.FIELD]["work_items"][0]["spans"][0]["spatial_Fortran_interval"][0] += 1
        with self.assertRaises(partials.ReviewError):
            self.replay(history + [self.event(altered, 5)])
        # Neither an unresolved nor a rejected physical attempt frees admission.
        for tail in ([], [{"event": "submission_not_accepted", "batch_id": "batch_0004", "evidence": "fixture"}]):
            account = self.replay(self.rows + [self.event(batch)] + tail)
            self.assertEqual(account["additional_remainder_durable_bytes"], 2368108032)
            with self.metadata(), self.assertRaises((accounting.AccountingError, partials.ReviewError)):
                submit.admit_batch(account, self.manifest, self.batch(("work_0006",)), results=self.results)

    def test_closed_predecessor_and_fresh_fragment_success_are_checked_before_later_refill(self):
        batch = self.batch()
        first = self.documents[submit.BASE / self.plan["work_items"][0]["predecessor_status"]]
        first["remaining_spans"][0]["spatial_Fortran_interval"][0] += 1
        with self.metadata(), self.assertRaises(partials.ReviewError):
            submit.admit_batch(self.account, self.manifest, batch, results=self.results)
        first["remaining_spans"][0]["spatial_Fortran_interval"][0] -= 1
        history = self.rows + [self.event(batch), self.base.accepted("900", "batch_0004")]
        live = self.replay(history)
        fresh = submit.build_batch(self.authority, self.manifest, "workers", ["work_0010"], "normal",
                                    manual_refill=True, deadline_partial_review=self.review)
        with self.metadata():
            submit.admit_batch(live, self.manifest, fresh, results=self.results)
        closed = self.replay(history + [self.base.terminal("900_4", allocation_job_id="901")])
        record = self.close_remainder_record("work_0005", "901")
        with self.metadata():
            submit.admit_batch(closed, self.manifest, fresh, results=self.results)
        record["committed_spans"] = self.manifest["work_items"][4]["spans"]
        with self.metadata(), self.assertRaisesRegex(partials.ReviewError, "exact_fresh_spans"):
            submit.admit_batch(closed, self.manifest, fresh, results=self.results)
        failed = self.replay(history + [self.base.terminal("900_4", state="FAILED", allocation_job_id="901")])
        with self.metadata(), self.assertRaisesRegex(partials.ReviewError, "successfully_charged"):
            submit.admit_batch(failed, self.manifest, fresh, results=self.results)

    def test_global_slots_cpu_and_queue_include_remainders_and_unfinished_original_work(self):
        batch = self.batch()
        fresh = submit.build_batch(self.authority, self.manifest, "workers",
            [f"work_{index:04d}" for index in range(10, 17)], "normal",
            manual_refill=True, deadline_partial_review=self.review)
        full = self.replay(self.rows + [self.event(fresh), self.base.accepted("900", "batch_0004")])
        with self.metadata(), self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            submit.admit_batch(full, self.manifest, batch, results=self.results)
        insufficient = copy.deepcopy(self.account)
        insufficient["budget"]["cpu_seconds"] = (insufficient["actual_CPU_seconds"]
            + insufficient["outstanding_reserved_CPU_seconds"] + 28799)
        with self.metadata(), self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            submit.admit_batch(insufficient, self.manifest, batch, results=self.results)
        history = self.rows + [self.event(batch), self.base.accepted("900", "batch_0004")]
        account = self.replay(history)
        next_batch = self.batch(("work_0006",))
        good = "800_8|ep21-full-workers-001|RUNNING\n900_4|ep21-full-workers-001|PENDING\n"
        for text, valid in ((good, True), (good.splitlines()[0]+"\n", False),
                            (good+"999_4|ep21-full-workers-001|RUNNING\n", False)):
            runner = mock.Mock(return_value=SimpleNamespace(stdout=text))
            if valid:
                submit.require_manual_refill_queue(account, self.authority, next_batch, runner)
            else:
                with self.assertRaises(submit.Stop):
                    submit.require_manual_refill_queue(account, self.authority, next_batch, runner)
            self.assertEqual(runner.call_count, 1)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 57600)

    def test_full_plan_and_reservation_are_durable_before_one_scheduler_submission(self):
        batch = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "journal.jsonl"
            with accounting.locked_events(journal, create=True) as (rows, append):
                for row in self.rows:
                    append(row)
                calls = []
                def scheduler(command, **kwargs):
                    calls.append(command)
                    event = json.loads(journal.read_text().splitlines()[-1])
                    self.assertEqual(event[remainder.FIELD], self.plan)
                    self.assertEqual(event["allocations"], [{"logical_id": "work_0005",
                                      "array_index": 4, "reserved_CPU_seconds": 28800}])
                    account = self.replay(rows)
                    self.assertEqual(account["outstanding_reserved_CPU_seconds"], 57600)
                    self.assertEqual(account["additional_remainder_durable_bytes"], 2368108032)
                    return SimpleNamespace(returncode=0, stdout="900\n", stderr="")
                with self.metadata():
                    submit.submit_one(batch, rows, append, self.authority, self.manifest,
                                      runner=scheduler, results=self.results)
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.replay(rows)["jobs"]["900_4"]["logical_id"], "work_0005")

    def test_explicit_assembly_requires_all_three_charged_exact_remainders_and_all_original_work(self):
        batch = self.batch(remainder.WORK_IDS)
        history = self.rows + [self.event(batch), self.base.accepted("900", "batch_0004"),
                              self.base.terminal("800_8", allocation_job_id="801")]
        for work in remainder.WORK_IDS:
            index = remainder.KNOWN[work][0]
            raw = str(910+index)
            history.append(self.base.terminal(f"900_{index}", allocation_job_id=raw))
            self.close_remainder_record(work, raw)
        assembly = submit.build_remainder_assembly(self.authority, self.manifest, self.plan, "normal")
        self.assertEqual(assembly["command"][-1], str(submit.REMAINDER_ASSEMBLY_WRAPPER))
        self.assertNotIn(accounting.MANUAL_REFILL_FIELD, assembly)
        with self.metadata(), self.assertRaisesRegex(submit.Stop, "all_required_work_items"):
            submit.admit_batch(self.replay(history), self.manifest, assembly, results=self.results)
        # Source-free history closes every other original assignment in finite
        # eight-worker waves; this reaches the actual final assembly admission.
        number = 5
        for begin in range(10, 76, 8):
            selected = [f"work_{index:04d}" for index in range(begin, min(begin+8, 76))]
            fresh = submit.build_batch(self.authority, self.manifest, "workers", selected, "normal",
                                        manual_refill=True, deadline_partial_review=self.review)
            parent = str(2000+number)
            history += [self.event(fresh, number), self.base.accepted(parent, f"batch_{number:04d}")]
            history.extend(self.base.terminal(f"{parent}_{int(work[5:])-1}",
                allocation_job_id=str(10000+int(work[5:]))) for work in selected)
            number += 1
        account = self.replay(history)
        with self.metadata():
            submit.admit_batch(account, self.manifest, assembly, results=self.results)
            ordinary = submit.build_batch(self.authority, self.manifest, "assembly", (), "normal")
            with self.assertRaisesRegex(submit.Stop, "failed_parent"):
                submit.admit_batch(account, self.manifest, ordinary, results=self.results)
        updated = self.replay(history + [self.event(assembly, number), self.base.accepted("3000", f"batch_{number:04d}")])
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"], 14400)
        self.assertEqual(updated["additional_remainder_durable_bytes"], 2368108032)
        for state in (None, "FAILED"):
            changed = copy.deepcopy(account)
            terminal = changed["jobs"]["900_4"]["terminal"]
            if state is None:
                changed["jobs"]["900_4"]["terminal"] = None
            else:
                terminal.update(state=state, exit_code="2:0")
            with self.metadata(), self.assertRaises((accounting.AccountingError, partials.ReviewError)):
                submit.admit_batch(changed, self.manifest, assembly, results=self.results)

    def test_same_physical_admission_contract_selects_remainder_array_id_without_ambiguity(self):
        from contextlib import nullcontext
        batch = self.batch()
        rows = self.rows + [self.event(batch), self.base.accepted("900", "batch_0004")]
        allocation = {"job_id": "901", "allocated_cpus": 1, "time_limit_seconds": 28800}
        with mock.patch.object(accounting, "locked_events", return_value=nullcontext((rows, None))), \
                mock.patch.dict(submit.os.environ, {"SLURM_ARRAY_JOB_ID": "900", "SLURM_ARRAY_TASK_ID": "4"}):
            submit.contract.admit_registered_allocation(self.authority, self.manifest, "work_0005", allocation)
        with mock.patch.object(accounting, "locked_events", return_value=nullcontext((rows, None))), \
                mock.patch.dict(submit.os.environ, {"SLURM_ARRAY_JOB_ID": "46839670", "SLURM_ARRAY_TASK_ID": "4"}), \
                self.assertRaisesRegex(submit.Stop, "unregistered_or_unfunded"):
            submit.contract.admit_registered_allocation(self.authority, self.manifest, "work_0005", allocation)


if __name__ == "__main__":
    unittest.main()
