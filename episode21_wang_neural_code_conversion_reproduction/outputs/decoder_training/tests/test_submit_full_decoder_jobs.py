"""Source-free finite-batch checks; run only in coordinated Slurm.

All scheduler replies, resource plans and task outcomes below are fixtures.
No real authority, manifest, feature, neural array or scheduler is read.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import submit_full_decoder_jobs as submit


class FullDecoderSubmissionTests(unittest.TestCase):
    def plans(self, count=80, layer=False):
        cap = {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
               "durable_bytes": 1_000_000, "scratch_bytes": 2_000_000_000}
        manifest = {"work_items": [{"work_item_id": f"work_{i + 1:04d}", "resources": dict(cap),
                     "participant": "sub-01", "spans": [], "units_requested": 1} for i in range(count)]}
        authority = {"cpu_budget_mode": accounting.MODE,
                     "included_layer_job": ({"job_id": "99", "reserved_CPU_seconds": 86400,
                                              "durable_bytes": 25_000_000_000} if layer else None),
                     "state_build_resources": {**cap, "minutes": 240},
                     "assembly_resources": {**cap, "minutes": 240},
                     "resources": {"maximum_concurrent_jobs": 8,
                                   "maximum_allocated_CPU_seconds": 500 * 3600,
                                   "durable_bytes": 500_000_000_000}}
        return authority, manifest

    def rows(self, authority, manifest):
        return [{"event": "budget_opened", "budget": accounting.budget_definition(authority)},
                {"event": "manifest_bound", "authority": copy.deepcopy(authority), "manifest": copy.deepcopy(manifest)}]

    def intent(self, authority, manifest, phase="states", selected=(), number=1):
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", "phase": phase,
                "allocations": accounting.allocations_for(authority, manifest, phase, selected)}

    def accepted(self, job="101", batch="batch_0001"):
        return {"event": "submission_accepted", "batch_id": batch, "job_id": job}

    def terminal(self, job="101", elapsed=60, state="COMPLETED", allocation_job_id=None):
        return {"event": "terminal_charge", "job_id": job, "allocation_job_id": allocation_job_id or job,
                "allocated_cpus": 1, "elapsed_seconds": elapsed, "state": state,
                "exit_code": "0:0" if state == "COMPLETED" else "2:0",
                "observed_utc": "2026-10-06T12:00:00Z", "evidence": "fixture: exact scheduler parent observation"}

    def refill_history(self, authority, manifest, closed=1, count=8):
        rows = self.rows(authority, manifest) + [self.intent(authority, manifest), self.accepted(), self.terminal(),
                self.intent(authority, manifest, "workers", [f"work_{i + 1:04d}" for i in range(count)], number=2),
                self.accepted("200", "batch_0002")]
        rows.extend(self.terminal(f"200_{i}", allocation_job_id=str(300 + i)) for i in range(closed))
        return rows

    def check_closed_fixture_results(self, account, manifest, *, overrides=None, read_jobs=None):
        """Exercise real closeout validation using only source-free fixture records."""
        def reader(path):
            logical = "states" if path.name == "state_parent_status.json" else path.parent.name
            job = next(job for job in account["jobs"].values() if job["logical_id"] == logical)
            if read_jobs is not None:
                read_jobs.append(job["job_id"])
            record = {"allocation": {"job_id": job["terminal"]["allocation_job_id"]}}
            if logical == "states":
                record.update(status="complete_four_participant_training_states_only",
                              participants_completed=list(submit.state.NEW_PARTICIPANTS), participants_remaining=[])
            else:
                item = next(item for item in manifest["work_items"] if item["work_item_id"] == logical)
                record.update(status="complete_work_item_only", work_item_id=logical,
                              participant=item["participant"], resources=item["resources"],
                              requested_spans=item["spans"], committed_spans=item["spans"],
                              units_requested=item["units_requested"], completed_units=item["units_requested"],
                              remaining_spans=[], completed_units_is_lower_bound=False, closeout_errors=[])
            record.update((overrides or {}).get(logical, {}))
            return record
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(Path, "is_symlink", return_value=False):
            submit.require_previous_results(account, manifest, reader=reader)

    def test_manual_refill_is_explicit_workers_only_and_default_still_waits_for_every_job(self):
        authority, manifest = self.plans()
        account = accounting.replay(self.refill_history(authority, manifest), authority, manifest)
        default = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal")
        refill = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal", manual_refill=True)
        self.assertNotIn(accounting.MANUAL_REFILL_FIELD, default)
        self.assertEqual(refill["allocations"], default["allocations"])
        self.assertEqual(refill["command"][:-1], default["command"][:-1])
        self.assertEqual(refill["command"][-1], str(submit.REFILL_WRAPPER))
        self.assertEqual(default["command"][-1], str(submit.BATCH_ROOT / "full_workers_001.sbatch"))
        with self.assertRaisesRegex(submit.Stop, "terminal_accounting_missing"):
            submit.admit_batch(account, manifest, default, results=self.check_closed_fixture_results)
        reads = []
        submit.admit_batch(account, manifest, refill,
                           results=lambda a, m: self.check_closed_fixture_results(a, m, read_jobs=reads))
        self.assertEqual(reads, ["101", "200_0"])
        for phase in ("states", "assembly"):
            with self.subTest(phase=phase), self.assertRaisesRegex(submit.Stop, "ordinary_workers"):
                submit.build_batch(authority, manifest, phase, (), "normal", manual_refill=True)

    def test_manual_refill_enforces_global_cross_array_slots_in_admission_and_replay(self):
        authority, manifest = self.plans()
        history = self.refill_history(authority, manifest, closed=2)
        refill = submit.build_batch(authority, manifest, "workers", ["work_0009", "work_0010"], "normal", manual_refill=True)
        account = accounting.replay(history, authority, manifest)
        submit.admit_batch(account, manifest, refill, results=self.check_closed_fixture_results)
        history += [{"event": "submission_intent", "batch_id": "batch_0003", **refill},
                    self.accepted("500", "batch_0003")]
        account = accounting.replay(history, authority, manifest)
        self.assertEqual(sum(job["terminal"] is None for job in account["jobs"].values()), 8)
        extra = submit.build_batch(authority, manifest, "workers", ["work_0011"], "normal", manual_refill=True)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            submit.admit_batch(account, manifest, extra, results=self.check_closed_fixture_results)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            accounting.replay(history + [{"event": "submission_intent", "batch_id": "batch_0004", **extra}],
                              authority, manifest)
        authority["resources"]["maximum_concurrent_jobs"] = 2
        account = accounting.replay(self.refill_history(authority, manifest, closed=1, count=2), authority, manifest)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            submit.admit_batch(account, manifest, refill, results=self.check_closed_fixture_results)

    def test_manual_refill_queue_requires_exact_known_live_workers_once(self):
        authority, manifest = self.plans()
        account = accounting.replay(self.refill_history(authority, manifest, closed=6), authority, manifest)
        batch = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal", manual_refill=True)
        good = "200_6|ep21-full-workers-001|RUNNING\n200_7|ep21-full-workers-001|PENDING\n"
        runner = mock.Mock(return_value=SimpleNamespace(stdout=good + "900|unrelated|RUNNING\n"))
        submit.require_manual_refill_queue(account, authority, batch, runner)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[0], ["squeue", "--me", "--array", "--noheader", "--format=%i|%j|%T"])
        for text, error in ((good.splitlines()[0] + "\n", "missing_reserved_worker"),
                            (good + "901|ep21-unknown|RUNNING\n", "unregistered_duplicate_or_unready"),
                            (good + "200_6|ep21-full-workers-001|RUNNING\n", "unregistered_duplicate_or_unready"),
                            (good.replace("200_7|ep21-full-workers-001", "200_7|renamed"), "unregistered_duplicate_or_unready"),
                            (good.replace("PENDING", "COMPLETED"), "unregistered_duplicate_or_unready"),
                            (good.replace("PENDING", "SUSPENDED"), "unregistered_duplicate_or_unready"),
                            (good + "200_0|ep21-full-workers-001|COMPLETING\n", "unregistered_duplicate_or_unready"),
                            (good + "200_0|renamed|COMPLETING\n", "unregistered_duplicate_or_unready")):
            with self.subTest(text=text), self.assertRaisesRegex(submit.Stop, error):
                submit.require_manual_refill_queue(account, authority, batch,
                        mock.Mock(return_value=SimpleNamespace(stdout=text)))
        # Absence above cannot release either slot or its full CPU reservation.
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 2 * 8 * 3600)

    def test_manual_refill_refuses_failed_partial_missing_or_uncertain_closed_results(self):
        authority, manifest = self.plans()
        history = self.refill_history(authority, manifest, closed=2)
        batch = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal", manual_refill=True)
        failed = copy.deepcopy(history)
        failed[-1].update(state="FAILED", exit_code="2:0")
        with self.assertRaisesRegex(submit.Stop, "failed_parent"):
            submit.admit_batch(accounting.replay(failed, authority, manifest), manifest, batch,
                               results=self.check_closed_fixture_results)
        account = accounting.replay(history, authority, manifest)
        for change in ({"status": "partial_work_item_only"}, {"completed_units": 0},
                       {"remaining_spans": [{"unfitted": True}]}, {"completed_units_is_lower_bound": True},
                       {"uncertain_publication": {}}, {"allocation": {"job_id": "999"}},
                       {"closeout_errors": [{"affects_fragment": True}]}):
            with self.subTest(change=change), self.assertRaises(submit.Stop):
                submit.admit_batch(account, manifest, batch,
                        results=lambda a, m: self.check_closed_fixture_results(a, m, overrides={"work_0001": change}))
        with mock.patch.object(Path, "is_file", return_value=False), self.assertRaisesRegex(submit.Stop, "result_missing"):
            submit.admit_batch(account, manifest, batch)
        duplicate = submit.build_batch(authority, manifest, "workers", ["work_0001"], "normal", manual_refill=True)
        with self.assertRaisesRegex(submit.Stop, "submitted_twice"):
            submit.admit_batch(account, manifest, duplicate, results=self.check_closed_fixture_results)

    def test_manual_refill_rejects_nonworker_prerequisites_and_unreconciled_submissions(self):
        authority, manifest = self.plans(layer=True)
        history = self.refill_history(authority, manifest)
        batch = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal", manual_refill=True)
        with self.assertRaisesRegex(accounting.AccountingError, "known_workers"):
            submit.admit_batch(accounting.replay(history, authority, manifest), manifest, batch)
        history.append(self.terminal("99"))
        no_state_closeout = [row for row in history if not (row["event"] == "terminal_charge" and row["job_id"] == "101")]
        with self.assertRaisesRegex(accounting.AccountingError, "completed_states"):
            submit.admit_batch(accounting.replay(no_state_closeout, authority, manifest), manifest, batch)
        initial = self.rows(authority, manifest) + [self.intent(authority, manifest), self.accepted(),
                                                   self.terminal(), self.terminal("99")]
        with self.assertRaisesRegex(accounting.AccountingError, "successfully_closed_worker"):
            submit.admit_batch(accounting.replay(initial, authority, manifest), manifest, batch)
        assembly = history + [self.intent(authority, manifest, "assembly", number=3), self.accepted("600", "batch_0003")]
        with self.assertRaisesRegex(accounting.AccountingError, "no_assembly"):
            submit.admit_batch(accounting.replay(assembly, authority, manifest), manifest, batch)
        uncertain = history + [self.intent(authority, manifest, "workers", ["work_0010"], number=3)]
        with self.assertRaisesRegex(accounting.AccountingError, "uncertain_submission"):
            submit.admit_batch(accounting.replay(uncertain, authority, manifest), manifest, batch)
        rejected = uncertain + [{"event": "submission_not_accepted", "batch_id": "batch_0003", "evidence": "fixture"}]
        with self.assertRaisesRegex(accounting.AccountingError, "rejected_attempt"):
            submit.admit_batch(accounting.replay(rejected, authority, manifest), manifest, batch)

    def test_manual_refill_intent_is_durable_before_submission_and_keeps_full_budget_reservations(self):
        authority, manifest = self.plans()
        original_authority, original_manifest = copy.deepcopy(authority), copy.deepcopy(manifest)
        batch = submit.build_batch(authority, manifest, "workers", ["work_0009"], "normal",
                                   constraint="CPU_GEN:SIE", manual_refill=True)
        history = self.refill_history(authority, manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for row in history:
                    append(row)
                def scheduler(command, **kwargs):
                    intent = json.loads(path.read_text().splitlines()[-1])
                    self.assertIs(intent[accounting.MANUAL_REFILL_FIELD], True)
                    self.assertEqual(intent["event"], "submission_intent")
                    self.assertEqual(intent["command"], command)
                    self.assertIn("--array=8%1", command)
                    self.assertEqual(command.count("--constraint=CPU_GEN:SIE"), 1)
                    reserved = accounting.replay(rows, authority, manifest)
                    self.assertEqual(reserved["outstanding_reserved_CPU_seconds"], 8 * 8 * 3600)
                    return SimpleNamespace(returncode=0, stdout="500\n", stderr="")
                result = submit.submit_one(batch, rows, append, authority, manifest, runner=scheduler,
                                           results=self.check_closed_fixture_results)
                account = accounting.replay(rows, authority, manifest)
                self.assertFalse(result["automatic_successor"])
                self.assertEqual(account["actual_CPU_seconds"], 120)
                self.assertEqual(account["outstanding_reserved_CPU_seconds"], 8 * 8 * 3600)
                self.assertEqual(account["budget"]["cpu_seconds"], 500 * 3600)
                with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
                    accounting.admit_reservation(account, 500 * 3600 - 8 * 8 * 3600 - 119)
        self.assertEqual(authority, original_authority)
        self.assertEqual(manifest, original_manifest)

    def test_subset_requests_preserve_global_indices_and_eight_job_throttle(self):
        authority, manifest = self.plans()
        ids = [f"work_{i:04d}" for i in range(2, 22, 2)]
        batch = submit.build_batch(authority, manifest, "workers", ids, "russpold,normal")
        self.assertIn("--array=1,3,5,7,9,11,13,15,17,19%8", batch["command"])
        self.assertEqual(sum(item["reserved_CPU_seconds"] for item in batch["allocations"]), 80 * 3600)
        self.assertGreater(len(manifest["work_items"]) * 8, 500)  # Full target remains assigned.
        authority["resources"]["maximum_concurrent_jobs"] = 3
        self.assertIn("--array=1,3,5,7,9,11,13,15,17,19%3",
                      submit.build_batch(authority, manifest, "workers", ids, "normal")["command"])

    def test_fixed_resources_order_and_no_preemptible_partition(self):
        authority, manifest = self.plans()
        manifest["work_items"][0]["resources"]["minutes"] = 240
        with self.assertRaisesRegex(submit.Stop, "fixed_wrapper_limits"):
            submit.build_batch(authority, manifest, "workers", ["work_0001"], "normal")
        manifest["work_items"].reverse()
        with self.assertRaisesRegex(accounting.AccountingError, "canonical_ordered"):
            submit.build_batch(authority, manifest, "states", (), "normal")
        with self.assertRaisesRegex(submit.Stop, "preemptible_owners"):
            submit.build_batch(authority, manifest, "states", (), "owners")

    def test_optional_CPU_constraint_preserves_assignments_resources_and_throttle(self):
        authority, manifest = self.plans()
        ids = [f"work_{i:04d}" for i in range(2, 22, 2)]
        default = submit.build_batch(authority, manifest, "workers", ids, "russpold,normal")
        constrained = submit.build_batch(authority, manifest, "workers", ids, "russpold,normal",
                                         constraint="CPU_GEN:SIE")
        flag = "--constraint=CPU_GEN:SIE"
        self.assertFalse(any(value.startswith("--constraint") for value in default["command"]))
        self.assertEqual(constrained["command"].count(flag), 1)
        self.assertEqual([value for value in constrained["command"] if value != flag], default["command"])
        self.assertEqual(constrained["allocations"], default["allocations"])
        self.assertEqual(sum(item["reserved_CPU_seconds"] for item in constrained["allocations"]), 80 * 3600)
        for value in ("--cpus-per-task=1", "--mem=7G", "--time=480",
                      "--array=1,3,5,7,9,11,13,15,17,19%8"):
            self.assertIn(value, constrained["command"])

    def test_CPU_constraint_rejects_other_features_and_nonworker_phases(self):
        authority, manifest = self.plans()
        for constraint in ("", "CPU_GEN:SKX", "CPU_GEN:SIE|CPU_GEN:SKX", "CPU_GEN:SIE --mem=14G", True):
            with self.subTest(constraint=constraint), self.assertRaisesRegex(submit.Stop, "only_verified_CPU"):
                submit.build_batch(authority, manifest, "workers", ["work_0001"], "normal",
                                   constraint=constraint)
        for phase in ("states", "assembly"):
            with self.subTest(phase=phase), self.assertRaisesRegex(submit.Stop, "ordinary_workers"):
                submit.build_batch(authority, manifest, phase, (), "normal", constraint="CPU_GEN:SIE")

    def test_CPU_constraint_is_recorded_before_submission_without_changing_budget(self):
        authority, manifest = self.plans()
        batch = submit.build_batch(authority, manifest, "workers", ["work_0001"], "normal",
                                   constraint="CPU_GEN:SIE")
        history = self.rows(authority, manifest) + [self.intent(authority, manifest),
                                                   self.accepted(), self.terminal()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for row in history:
                    append(row)
                def scheduler(command, **kwargs):
                    intent = json.loads(path.read_text().splitlines()[-1])
                    self.assertEqual(intent["event"], "submission_intent")
                    self.assertEqual(intent["command"], command)
                    self.assertEqual(command.count("--constraint=CPU_GEN:SIE"), 1)
                    return SimpleNamespace(returncode=0, stdout="201\n", stderr="")
                submit.submit_one(batch, rows, append, authority, manifest,
                                  runner=scheduler, results=lambda *_: None)
                account = accounting.replay(rows, authority, manifest)
                self.assertEqual(account["actual_CPU_seconds"], 60)
                self.assertEqual(account["outstanding_reserved_CPU_seconds"], 8 * 3600)
                self.assertEqual(account["budget"]["cpu_seconds"], 500 * 3600)

    def test_first_layer_reservation_replaced_by_actual_charge(self):
        authority, manifest = self.plans(layer=True)
        rows = self.rows(authority, manifest)
        account = accounting.replay(rows, authority, manifest)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 86400)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            accounting.admit_reservation(account, 477 * 3600)
        rows.append(self.terminal("99", elapsed=3600))
        account = accounting.replay(rows, authority, manifest)
        self.assertEqual((account["actual_CPU_seconds"], account["outstanding_reserved_CPU_seconds"]), (3600, 0))
        accounting.admit_reservation(account, 499 * 3600)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            accounting.admit_reservation(account, 499 * 3600 + 1)

    def test_each_array_element_retains_full_reservation_until_its_terminal_record(self):
        authority, manifest = self.plans()
        rows = self.rows(authority, manifest) + [self.intent(authority, manifest, "workers", ["work_0001", "work_0002"]),
                                                self.accepted("200")]
        account = accounting.replay(rows, authority, manifest)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 16 * 3600)
        rows.append(self.terminal("200_0", 100, allocation_job_id="201"))
        account = accounting.replay(rows, authority, manifest)
        self.assertEqual((account["actual_CPU_seconds"], account["outstanding_reserved_CPU_seconds"]), (100, 8 * 3600))
        with self.assertRaisesRegex(accounting.AccountingError, "charged_twice"):
            accounting.replay(rows + [self.terminal("200_1", 100, allocation_job_id="201")], authority, manifest)
        with self.assertRaisesRegex(accounting.AccountingError, "charged_once"):
            accounting.replay(rows + [rows[-1]], authority, manifest)

    def test_immutable_binding_duplicate_work_and_reservation_tampering_stop(self):
        authority, manifest = self.plans()
        rows = self.rows(authority, manifest)
        changed = copy.deepcopy(manifest)
        changed["work_items"][0]["units_requested"] = 2
        with self.assertRaisesRegex(accounting.AccountingError, "manifest_is_immutable"):
            accounting.replay(rows, authority, changed)
        changed_auth = copy.deepcopy(authority)
        changed_auth["resources"]["maximum_concurrent_jobs"] = 7
        with self.assertRaisesRegex(accounting.AccountingError, "authority_is_immutable"):
            accounting.replay(rows, changed_auth)
        intent = self.intent(authority, manifest)
        wrong = copy.deepcopy(intent)
        wrong["allocations"][0]["reserved_CPU_seconds"] -= 1
        with self.assertRaisesRegex(accounting.AccountingError, "intent_reservations"):
            accounting.replay(rows + [wrong], authority, manifest)
        with self.assertRaisesRegex(accounting.AccountingError, "submitted_twice"):
            accounting.replay(rows + [intent, self.intent(authority, manifest, number=2)], authority, manifest)

    def test_uncertain_submission_keeps_reservation_and_never_retries(self):
        authority, manifest = self.plans()
        batch = submit.build_batch(authority, manifest, "states", (), "normal")
        for response in [subprocess.TimeoutExpired("sbatch", 30),
                         SimpleNamespace(returncode=1, stdout="", stderr="rejected"),
                         SimpleNamespace(returncode=0, stdout="unknown", stderr="")]:
            with self.subTest(response=str(response)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "submission.jsonl"
                with accounting.locked_events(path, create=True) as (rows, append):
                    for row in self.rows(authority, manifest):
                        append(row)
                    runner = mock.Mock(side_effect=response) if isinstance(response, Exception) else mock.Mock(return_value=response)
                    with self.assertRaisesRegex(submit.Stop, "uncertain"):
                        submit.submit_one(batch, rows, append, authority, manifest, runner=runner, results=lambda *_: None)
                    self.assertEqual(runner.call_count, 1)
                    account = accounting.replay(rows, authority, manifest)
                    self.assertEqual(account["outstanding_reserved_CPU_seconds"], 4 * 3600)
                    with self.assertRaisesRegex(accounting.AccountingError, "uncertain_submission"):
                        accounting.admit_reservation(account, 1)
                    self.assertEqual(rows[-1]["event"], "submission_uncertain")

    def test_one_submission_intent_precedes_call_and_strips_ambient_SBATCH(self):
        authority, manifest = self.plans()
        batch = submit.build_batch(authority, manifest, "states", (), "normal")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(submit.os.environ, {"SBATCH_GPUS": "4"}):
            path = Path(directory) / "submission.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for row in self.rows(authority, manifest):
                    append(row)
                def scheduler(command, **kwargs):
                    self.assertEqual(json.loads(path.read_text().splitlines()[-1])["event"], "submission_intent")
                    self.assertFalse(any(key.startswith("SBATCH_") for key in kwargs["env"]))
                    return SimpleNamespace(returncode=0, stdout="101\n", stderr="")
                result = submit.submit_one(batch, rows, append, authority, manifest, runner=scheduler, results=lambda *_: None)
                self.assertEqual(result["job_id"], "101")
                self.assertEqual(rows[-1]["event"], "submission_accepted")
                self.assertFalse(result["automatic_successor"])

    def test_resolution_accepts_once_or_stops_after_confirmed_nonacceptance(self):
        authority, manifest = self.plans()
        rows = self.rows(authority, manifest) + [self.intent(authority, manifest)]
        account = accounting.replay(rows + [self.accepted()], authority, manifest)
        self.assertFalse(account["uncertain_batches"])
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 14400)
        rejected = {"event": "submission_not_accepted", "batch_id": "batch_0001", "evidence": "operator confirmed no job"}
        account = accounting.replay(rows + [rejected], authority, manifest)
        with self.assertRaisesRegex(accounting.AccountingError, "rejected_attempt"):
            accounting.admit_reservation(account, 1)
        with self.assertRaisesRegex(accounting.AccountingError, "one_explicit"):
            accounting.replay(rows + [rejected, self.accepted()], authority, manifest)

    def test_failure_is_charged_but_missing_or_failed_results_stop_next_batch(self):
        authority, manifest = self.plans()
        rows = self.rows(authority, manifest) + [self.intent(authority, manifest), self.accepted()]
        account = accounting.replay(rows + [self.terminal(state="FAILED")], authority, manifest)
        self.assertEqual(account["actual_CPU_seconds"], 60)
        with self.assertRaisesRegex(submit.Stop, "failed_parent"):
            submit.require_previous_results(account, manifest)
        account = accounting.replay(rows + [self.terminal()], authority, manifest)
        with mock.patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(submit.Stop, "result_missing"):
                submit.require_previous_results(account, manifest)
        with self.assertRaisesRegex(submit.Stop, "terminal_accounting_missing"):
            submit.require_previous_results(accounting.replay(rows, authority, manifest), manifest)

    def test_live_EP21_job_blocks_queue_without_touching_unrelated_job(self):
        queue = mock.Mock(return_value=SimpleNamespace(stdout="123|ep21-layer-001|PENDING\n"))
        with self.assertRaisesRegex(submit.Stop, "existing_EP21_job"):
            submit.require_no_live_episode_job(queue)
        self.assertEqual(queue.call_count, 1)
        submit.require_no_live_episode_job(mock.Mock(return_value=SimpleNamespace(stdout="456|unrelated|RUNNING\n")))

    def test_worker_read_waits_for_submission_lock_and_torn_history_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.jsonl"
            path.write_text('{"event":"fixture"}\n')
            with mock.patch.object(accounting.fcntl, "flock") as lock, \
                    mock.patch.object(accounting.signal, "setitimer", wraps=accounting.signal.setitimer) as timer:
                with accounting.locked_events(path, shared=True):
                    pass
                self.assertEqual(lock.call_args.args[1], accounting.fcntl.LOCK_SH)
                self.assertEqual([call.args[1] for call in timer.call_args_list], [45, 0])
            with mock.patch.object(accounting.fcntl, "flock") as lock:
                with accounting.locked_events(path):
                    pass
                self.assertEqual(lock.call_args.args[1], accounting.fcntl.LOCK_EX | accounting.fcntl.LOCK_NB)
            path.write_text('{"event":')
            with self.assertRaisesRegex(accounting.AccountingError, "incomplete_accounting"):
                with accounting.locked_events(path):
                    pass

    def test_workers_and_assembly_cannot_skip_required_phases(self):
        authority, manifest = self.plans()
        account = accounting.replay(self.rows(authority, manifest), authority, manifest)
        workers = submit.build_batch(authority, manifest, "workers", ["work_0001"], "normal")
        with self.assertRaisesRegex(submit.Stop, "states_required"):
            submit.admit_batch(account, manifest, workers, results=lambda *_: None)
        rows = self.rows(authority, manifest) + [self.intent(authority, manifest), self.accepted(), self.terminal()]
        account = accounting.replay(rows, authority, manifest)
        assembly = submit.build_batch(authority, manifest, "assembly", (), "normal")
        with self.assertRaisesRegex(submit.Stop, "all_required_work_items"):
            submit.admit_batch(account, manifest, assembly, results=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
