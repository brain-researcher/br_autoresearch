"""Four focused I/O remainder admission checks; coordinator allocation only.

Uses fabricated closed metadata and temporary empty output directories. No
scientific payload, original failed HDF, scheduler or real journal is opened.
"""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewed_io_recovery as recovery
import train_decoder_remainder as worker
import test_reviewed_io_recovery as fixtures

POLICY = {"effective_scratch_bytes": 1012826112, "parallelism_amendment_id": "parallelism_amendment"}


class StopBeforePayload(RuntimeError):
    pass


class IORecoveryWorkerTests(unittest.TestCase):
    def fixture(self, base):
        review, manifest, jobs, records, reports, quarantine = fixtures.fixture(base_directory=str(base))
        item = review["work_items"][0]
        _, job = fixtures.completed_record(review, item["work_item_id"], base_directory=str(base))
        job["terminal"] = None
        batch = {**fixtures.batch(review, quarantine), "accepted_id": "9000"}
        account = {"jobs": {**jobs, job["job_id"]: job}, "batches": {job["batch_id"]: batch}}
        return review, manifest, records, reports, account, item

    def write_metadata(self, base, review, records, reports):
        documents = {recovery.REVIEW_RELATIVE: review, **reports}
        documents.update({item["predecessor_status"]: records[item["work_item_id"]]
                          for item in review["work_items"]})
        for relative, record in documents.items():
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))

    def test_fixed_closed_metadata_admission_does_not_open_failed_fragments(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            review, manifest, records, reports, account, _ = self.fixture(base)
            self.write_metadata(base, review, records, reports)
            with mock.patch.object(worker.prefix, "BASE", base), \
                    mock.patch.object(worker.accounting, "locked_events", return_value=nullcontext(([], None))), \
                    mock.patch.object(worker.accounting, "replay", return_value=account), \
                    mock.patch.object(worker.prefix, "load_libraries", side_effect=AssertionError("no scientific imports")):
                returned, items, observed_account = worker.admit_io_recovery(base / recovery.REVIEW_RELATIVE, {}, manifest)
                self.assertEqual(returned, review)
                self.assertEqual(tuple(items), recovery.WORK_IDS)
                self.assertIs(observed_account, account)
                self.assertTrue(all(not (base / item["predecessor_fragment"]).exists()
                                    for item in review["work_items"]))
                damaged = deepcopy(reports)
                report_path = review["work_items"][0]["inspection"]["report"]
                damaged[report_path]["fragments"][0]["inspection_complete"] = False
                (base / report_path).write_text(json.dumps(damaged[report_path]))
                with self.assertRaisesRegex(ValueError, "complete_consistent_exact_prefix"):
                    worker.admit_io_recovery(base / recovery.REVIEW_RELATIVE, {}, manifest)
                with self.assertRaisesRegex(RuntimeError, "fixed_exact_IO_recovery"):
                    worker.admit_io_recovery(base / "other.json", {}, manifest)

    def test_current_array_and_accepted_effective_cap_are_required(self):
        review, manifest, _, _, account, item = self.fixture(Path("/fixture/ep21"))
        allocation = {"job_id": "9001", "allocated_cpus": 1, "memory_gib": 7,
                      "gpus": 0, "time_limit_seconds": 28800}
        environment = {"SLURM_ARRAY_JOB_ID": "9000", "SLURM_ARRAY_TASK_ID": "31"}
        with mock.patch.dict(worker.os.environ, environment), \
                mock.patch.object(worker.contract, "admit_registered_allocation", return_value=POLICY) as admit:
            job, registered, policy = worker.admit_io_recovery_allocation({}, manifest, review, account, item, allocation)
            self.assertEqual(job["job_id"], "9000_31")
            self.assertEqual(registered[recovery.FIELD], review)
            before = deepcopy(item["resources"])
            effective = worker.remainder_runtime_resources(manifest, item["work_item_id"], item["resources"], policy)
            self.assertEqual(effective["scratch_bytes"], 1012826112)
            self.assertEqual(item["resources"], before)
            self.assertEqual(before["scratch_bytes"], 2000000000)
            for bad in (None, {**POLICY, "effective_scratch_bytes": 2000000000},
                        {**POLICY, "parallelism_amendment_id": "other"}):
                admit.return_value = bad
                with self.assertRaisesRegex(RuntimeError, "accepted_IO_remainder_effective_scratch"):
                    worker.admit_io_recovery_allocation({}, manifest, review, account, item, allocation)
        with mock.patch.dict(worker.os.environ, {**environment, "SLURM_ARRAY_TASK_ID": "32"}), \
                mock.patch.object(worker.contract, "admit_registered_allocation") as forbidden:
            with self.assertRaisesRegex(RuntimeError, "exact_registered_IO_remainder"):
                worker.admit_io_recovery_allocation({}, manifest, review, account, item, allocation)
            forbidden.assert_not_called()
        # The existing legacy/serial caller passes None and keeps its resource map.
        self.assertEqual(worker.remainder_runtime_resources(manifest, item["work_item_id"], before, None), before)

    def setup_run(self, stack, base, *, existing_output=False, policy=POLICY):
        review, manifest, _, _, account, item = self.fixture(base)
        manifest_path = base / recovery.ROOT / "work_manifest.json"
        scratch = base / "node_local"
        scratch.mkdir()
        output = base / item["output_directory"]
        if existing_output:
            output.mkdir(parents=True)
        args = SimpleNamespace(authorization=str(base / "outputs/decoder_training_full_authorization.json"),
            manifest=str(manifest_path), remainder_plan=None, io_recovery_review=str(base / recovery.REVIEW_RELATIVE),
            work_item=item["work_item_id"], max_seconds=28500)
        allocation = {"job_id": "9001", "allocated_cpus": 1, "memory_gib": 7,
                      "gpus": 0, "time_limit_seconds": 28800}
        stack.enter_context(mock.patch.object(Path, "cwd", return_value=base))
        stack.enter_context(mock.patch.object(worker.prefix, "BASE", base))
        stack.enter_context(mock.patch.object(worker.contract, "MANIFEST", manifest_path))
        stack.enter_context(mock.patch.object(worker.state, "authorize_full", return_value={}))
        stack.enter_context(mock.patch.object(worker.prefix, "read_json", return_value=manifest))
        stack.enter_context(mock.patch.object(worker.contract, "validate_work_manifest"))
        stack.enter_context(mock.patch.object(worker, "admit_io_recovery", return_value=(review,
            {row["work_item_id"]: row for row in review["work_items"]}, account)))
        stack.enter_context(mock.patch.object(worker.contract, "verify_allocation", return_value=allocation))
        stack.enter_context(mock.patch.object(worker.contract, "admit_registered_allocation", return_value=policy))
        stack.enter_context(mock.patch.dict(worker.os.environ, {"SLURM_ARRAY_JOB_ID": "9000",
            "SLURM_ARRAY_TASK_ID": "31", "L_SCRATCH_JOB": str(scratch)}))
        budget = stack.enter_context(mock.patch.object(worker.contract, "WorkBudget", return_value=SimpleNamespace(seconds=28500)))
        metadata = stack.enter_context(mock.patch.object(worker.state, "admit_participant", return_value={}))
        shared = stack.enter_context(mock.patch.object(worker.state, "load_shared", side_effect=AssertionError("no payload")))
        libraries = stack.enter_context(mock.patch.object(worker.prefix, "load_libraries", side_effect=AssertionError("no numerical imports")))
        return args, item, allocation, output, budget, metadata, shared, libraries

    def test_fresh_status_and_budget_bind_review_without_changing_declared_resources(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            args, item, allocation, output, budget, metadata, shared, libraries = self.setup_run(stack, Path(temporary))
            captured = {}
            def publish(path, record, current_budget):
                captured.update(path=path, record=deepcopy(record))
                raise StopBeforePayload("stop after operational admission")
            stack.enter_context(mock.patch.object(worker.prefix, "publish_json", side_effect=publish))
            with self.assertRaisesRegex(StopBeforePayload, "operational admission"):
                worker.run(args)
            record = captured["record"]
            self.assertEqual(captured["path"], output / "requested_units.json")
            self.assertEqual(record["io_recovery_review"], args.io_recovery_review)
            self.assertNotIn("remainder_plan", record)
            self.assertEqual(record["resources"], item["resources"])
            self.assertEqual(record["effective_scratch_bytes"], 1012826112)
            self.assertEqual(record["parallelism_amendment_id"], "parallelism_amendment")
            self.assertEqual(record["requested_spans"], item["spans"])
            self.assertEqual(record["units_requested"], 232960)
            self.assertEqual(record["predecessor_status"], item["predecessor_status"])
            self.assertEqual(budget.call_args.args, (allocation, {**item["resources"], "scratch_bytes": 1012826112}, 28500))
            metadata.assert_called_once_with("sub-05")
            shared.assert_not_called()
            libraries.assert_not_called()

    def test_existing_output_and_unadmitted_policy_stop_before_training_access_and_cli_excludes_dual_routes(self):
        for existing, policy, message in ((True, POLICY, "remainder_output_must_be_fresh"),
                                         (False, None, "accepted_IO_remainder_effective_scratch")):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                args, _, _, _, _, metadata, shared, libraries = self.setup_run(
                    stack, Path(temporary), existing_output=existing, policy=policy)
                with self.assertRaisesRegex(RuntimeError, message):
                    worker.run(args)
                metadata.assert_not_called()
                shared.assert_not_called()
                libraries.assert_not_called()
        command = ["train_decoder_remainder.py", "--authorization", "auth", "--manifest", "manifest",
            "--work-item", "work_0032", "--remainder-plan", "deadline", "--io-recovery-review", "review"]
        with mock.patch.object(sys, "argv", command), mock.patch.object(worker, "run") as forbidden:
            with self.assertRaises(SystemExit):
                worker.main()
            forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
