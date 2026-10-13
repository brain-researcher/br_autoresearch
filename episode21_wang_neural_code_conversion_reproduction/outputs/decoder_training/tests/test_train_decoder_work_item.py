"""Focused generalized-worker checks; run only in approved substantive Slurm.

No source payloads, prior weights or protected outcomes are opened by fixtures.
They cover new assignment/participant dispatch and FC-tail behavior; previously
passed source arithmetic and grouped-publication checks are not repeated.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_decoder_work_item as worker
import plan_full_decoder_work as planner

contract, prefix = worker.contract, worker.prefix


class WorkItemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix.load_libraries()
        cls.np, cls.h5py = prefix.np, prefix.h5py

    def resources(self):
        return {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
                "durable_bytes": 8_000_000_000, "scratch_bytes": 2_000_000_000}

    def budget(self):
        return contract.WorkBudget({"job_end_epoch": time.time() + 28800}, self.resources())

    def timing(self):
        return {"normalization_correlation_RHS_and_exact500_solve_seconds": 0.,
                "select_solve_seconds": 0., "publication_seconds": 0.}

    def plan(self):
        retained = tuple(contract.coverage.retained_spans())
        plan = planner.plan_from_spans(retained)
        plan["retained"] = []  # The specific metadata reader is isolated below.
        caps = [item["resources"] for item in plan["work_items"]]
        caps += [plan["state_build_resources"], plan["assembly_resources"]]
        authority = {"work_manifest": contract.MANIFEST_RELATIVE, "reuse_completed_units": 51328,
                     "state_build_resources": plan["state_build_resources"],
                     "assembly_resources": plan["assembly_resources"],
                     "resources": {"worker_jobs": len(plan["work_items"]), "state_build_jobs": 1,
                         "assembly_jobs": 1, "maximum_concurrent_jobs": 8,
                         "maximum_allocated_CPU_seconds": sum(cap["minutes"] * 60 for cap in caps),
                         "durable_bytes": sum(cap["durable_bytes"] for cap in caps),
                         "peak_transient_bytes": 16_000_000_000}}
        return retained, plan, authority

    def test_consumed_authority_cannot_read_manifest_or_allocate(self):
        for old in (prefix.AUTH, worker.layer.prior.AUTH, worker.layer.AUTH):
            with mock.patch.object(prefix, "read_json") as read:
                with self.assertRaisesRegex(prefix.Stop, "separate_fixed_full_stage"):
                    worker.state.authorize_full(old)
                read.assert_not_called()
        with mock.patch.object(Path, "cwd", return_value=prefix.BASE), \
                mock.patch.object(worker.state, "authorize_full", side_effect=prefix.Stop("not approved")), \
                mock.patch.object(prefix, "read_json") as read, \
                mock.patch.object(worker, "verify_allocation") as allocation, \
                mock.patch.object(prefix, "load_libraries") as libraries:
            with self.assertRaisesRegex(prefix.Stop, "not approved"):
                worker.run(SimpleNamespace(authorization=worker.state.FULL_AUTH,
                           manifest=contract.MANIFEST, work_item="work_0001", max_seconds=None))
            read.assert_not_called()
            allocation.assert_not_called()
            libraries.assert_not_called()

    def test_full_coverage_retained_exclusion_and_aggregate_caps(self):
        retained, plan, authority = self.plan()
        with mock.patch.object(contract, "read_retained_coverage", return_value=retained):
            assignments = contract.validate_work_manifest(plan, authority)
            self.assertEqual(len(assignments), len(plan["work_items"]))
            self.assertEqual({item["participant"] for item in assignments.values()},
                             set(contract.coverage.PARTICIPANTS))
            self.assertEqual(sum(item["units_requested"] for item in assignments.values()), 74_255_112)
            wrong_count = deepcopy(plan)
            wrong_count["work_items"][0]["units_requested"] += 1
            with self.assertRaisesRegex(prefix.Stop, "assigned_unit_count"):
                contract.validate_work_manifest(wrong_count, authority)
            duplicate = deepcopy(plan)
            duplicate["work_items"].append(dict(duplicate["work_items"][0], work_item_id="work_duplicate"))
            with self.assertRaisesRegex(prefix.Stop, "coverage_gap_or_overlap"):
                contract.validate_work_manifest(duplicate, authority)
            for key in ("maximum_allocated_CPU_seconds", "durable_bytes", "peak_transient_bytes"):
                smaller = deepcopy(authority)
                smaller["resources"][key] -= 1
                with self.assertRaisesRegex(prefix.Stop, "aggregate_full_stage_caps"):
                    contract.validate_work_manifest(plan, smaller)
            changed = dict(authority, reuse_completed_units=51329)
            with self.assertRaisesRegex(prefix.Stop, "authorized_retained_units"):
                contract.validate_work_manifest(plan, changed)

    def test_terminal_partial_layer_preserves_exact_committed_rows(self):
        coverage = contract.coverage
        committed = [coverage.Span("sub-01", "conv1_1", 1, 0, 3)]
        remaining = [coverage.Span("sub-01", "conv1_1", 1, 3, 50176), *worker.layer.SPANS[1:]]
        report = {"schema_version": "ep21.decoder_layer_continuation.v1",
                  "status": "partial_layer_continuation_stopped", "allocation": {"job_id": "42"},
                  "participant": "sub-01", "layer": "conv1_1", "training_role": "subject_unique",
                  "training_images": 4712, "training_repetitions": 18843,
                  "shared_state_reference": str(prefix.OUTPUT / "shared_training_state.npz"),
                  "units_requested": 3161088, "retained_completed_units": 51328,
                  "fragment": "channels.h5", "remaining_interval_status": "exact_uncommitted_coverage",
                  "completed_units_is_lower_bound": False, "closeout_errors": [], "completed_units": 3,
                  "requested_spans": [span.record() for span in worker.layer.SPANS],
                  "committed_spans": [span.record() for span in committed],
                  "remaining_spans": [span.record() for span in remaining],
                  **{key: False for key in ("predictions_or_scores_computed", "tau_train_values_read",
                     "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized")}}
        entries = [
            {"kind": "retained_prefix", "status": "outputs/decoder_training/retained_prefix_001/prefix_status.json"},
            {"kind": "channel_continuation", "status": "outputs/decoder_training/continuation_001/continuation_status.json"},
            {"kind": "layer_continuation", "status": "outputs/decoder_training/layer_continuation_001/continuation_status.json",
             "closed_parent": {"job_id": "42", "state": "FAILED", "exit_code": "2:0", "observed_utc": "fixture"}},
        ]
        def read(path):
            return {"training_image_ids": ["fixture"]} if path.name == "training_row_binding.json" else report
        with mock.patch.object(contract, "_safe_file"), \
                mock.patch.object(worker.layer, "admit_completed"), \
                mock.patch.object(prefix, "read_json", side_effect=read):
            with mock.patch.object(Path, "exists", return_value=True):
                with self.assertRaisesRegex(prefix.Stop, "existing_layer_attempt"):
                    contract.read_retained_coverage(entries[:2])
            retained = contract.read_retained_coverage(entries)
            self.assertEqual(sum(span.count for span in retained), 51331)
            pending = planner.remaining_after(retained)
            current = next(span for span in pending if span.key == committed[0].key)
            self.assertEqual(current.start, 3)
            report["uncertain_publication"] = {"spatial_Fortran_interval": [3, 4]}
            with self.assertRaisesRegex(prefix.Stop, "uncertain_fragment"):
                contract.read_retained_coverage(entries)
            del report["uncertain_publication"]
            report["closeout_errors"] = [{"affects_fragment": True}]
            with self.assertRaisesRegex(prefix.Stop, "uncertain_fragment"):
                contract.read_retained_coverage(entries)

    def test_exact_partial_coverage_is_not_permission_to_retry_invalid_targets(self):
        entry = {"kind": "layer_continuation", "status": "fixture_status.json"}
        report = {"allocation": {"job_id": "42"}, "completed_units": 3,
                  "units_requested": 3161088, "error_type": "Stop", "error": "layer_deadline_reached"}
        approved = {"resume_partial_layer": {"job_id": "42", "error": "layer_deadline_reached",
                                            "cause": "resource_or_deadline"}}
        with mock.patch.object(prefix, "read_json", return_value=report):
            with self.assertRaisesRegex(prefix.Stop, "explicit_new_authority"):
                contract.validate_retained_continuation_causes([entry], {})
            contract.validate_retained_continuation_causes([entry], approved)
            report["error"] = "undefined_required_feature_training_scale"
            with self.assertRaisesRegex(prefix.Stop, "correction_or_scientific_amendment"):
                contract.validate_retained_continuation_causes([entry], approved)
            report["completed_units"] = report["units_requested"]
            contract.validate_retained_continuation_causes([entry], {})

    def test_scheduler_end_controls_budget_without_optional_environment_snapshot(self):
        end = datetime.fromtimestamp(38800).isoformat()
        text = ("JobId=42 JobState=RUNNING NumCPUs=1 NumNodes=1 NodeList=compute "
                f"AllocTRES=cpu=1,mem=7G TimeLimit=08:00:00 EndTime={end}")
        environment = {"SLURM_JOB_ID": "42", "SLURM_CPUS_PER_TASK": "1", "SLURM_MEM_PER_NODE": "7168",
                       **{name: "1" for name in prefix.THREADS}}
        with mock.patch.dict(contract.os.environ, environment, clear=True), \
                mock.patch.object(contract.socket, "gethostname", return_value="compute.domain"), \
                mock.patch.object(contract.time, "time", return_value=10000), \
                mock.patch.object(contract.subprocess, "run", return_value=SimpleNamespace(stdout=text)) as command:
            allocation = contract.verify_allocation(self.resources())
            budget = contract.WorkBudget(allocation, self.resources())
            self.assertEqual(budget.seconds, 28500)
            self.assertIs(budget.allocation, allocation)
            command.return_value = SimpleNamespace(stdout=text.replace("08:00:00", "09:00:00"))
            with self.assertRaisesRegex(prefix.Stop, "authorized_walltime"):
                contract.verify_allocation(self.resources())

    def test_FC_loader_reads_tail_and_preserves_exact_unit_identity(self):
        np = self.np
        span = contract.coverage.Span("sub-07", "fc8", 0, 64, 1000)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(prefix, "IMAGES", 3):
            root = Path(temporary)
            data = np.arange(3000, dtype=np.float32).reshape(3, 1000)
            valid = np.ones((3, 1000), dtype=np.uint8)
            valid[:, :64] = 0  # Excluded units cannot invalidate this requested tail.
            with self.h5py.File(root / "features.h5", "w") as source:
                source.create_dataset("fc8", data=data)
            with self.h5py.File(root / "finite_packed.h5", "w") as flags:
                flags.create_dataset("fc8", data=np.packbits(valid, axis=1, bitorder="little"))
            bank = worker.layer.stage_channel(root, root, span, self.budget())
            try:
                np.testing.assert_array_equal(bank[:, 64:], data[:, 64:])
            finally:
                bank._mmap.close()
                (root / "channel.float32").unlink()
            path = root / "fragment.h5"
            with self.h5py.File(path, "x") as container:
                store = worker.layer.ChannelStore(container, path,
                    contract.coverage.Span("sub-07", "fc8", 0, 64, 67), self.budget())
                np.testing.assert_array_equal(store.file["full_layer_Fortran_index"][:], [64, 65, 66])
                np.testing.assert_array_equal(store.file["full_layer_C_index"][:], [64, 65, 66])
                self.assertEqual(store.file.attrs["participant"], "sub-07")

    def test_exact500_branch_uses_native_order_and_genuine_FC_short_tail(self):
        np = self.np
        span = contract.coverage.Span("sub-03", "fc8", 0, 936, 1000)
        native = np.arange(500, dtype=np.int64) * 3
        bank = np.zeros((5, 1000), dtype=np.float32)
        solved = np.arange(501 * 64, dtype=np.float32).reshape(501, 64)
        prepared = {"branch": "source_exactly_500_native_order_no_selection", "solved": solved,
                    "means": np.zeros((1, 64), dtype=np.float32), "scales": np.ones((1, 64), dtype=np.float32)}
        store, budget = mock.Mock(), self.budget()
        with mock.patch.object(prefix, "prepare_layer_math", return_value=prepared) as prepare, \
                mock.patch.object(prefix, "solve_selected") as selection:
            worker.fit_span(bank, span, None, None, None, native, None, store, budget, self.timing())
        selection.assert_not_called()
        self.assertEqual(prepare.call_args.args[3].shape, (5, 64))
        self.assertTrue(prepare.call_args.args[3].flags.c_contiguous)
        begin, rows = store.commit.call_args.args
        self.assertEqual(begin, 936)
        self.assertEqual(len(rows), 64)
        np.testing.assert_array_equal(rows[63][0], native)
        np.testing.assert_array_equal(rows[63][1], solved[:500, 63])
        self.assertEqual(rows[63][2], solved[500, 63])
        # The last FC8 block in a whole unprefixed channel has 40 actual targets.
        shorter = contract.coverage.Span("sub-03", "fc8", 0, 960, 1000)
        prepared.update(solved=solved[:, :40], means=prepared["means"][:, :40], scales=prepared["scales"][:, :40])
        with mock.patch.object(prefix, "prepare_layer_math", return_value=prepared) as prepare:
            worker.fit_span(bank, shorter, None, None, None, native, None, store, budget, self.timing())
        self.assertEqual(prepare.call_args.args[3].shape, (5, 40))
        self.assertEqual(len(store.commit.call_args.args[1]), 40)

    def test_selected_branch_keeps_ranked_ids_and_commits_partial_block_on_stop(self):
        np = self.np
        span = contract.coverage.Span("sub-05", "conv5_1", 9, 128, 196)
        native = np.arange(501, dtype=np.int64) * 2
        selected = np.arange(499, -1, -1)
        bank = np.zeros((5, 196), dtype=np.float32)
        prepared = {"branch": "source_greater_than_500_with_native_ID_ties",
                    "right": np.zeros((64, 502)), "correlations": np.zeros((64, 501)),
                    "means": np.zeros((1, 64), dtype=np.float32), "scales": np.ones((1, 64), dtype=np.float32)}
        store, budget = mock.Mock(), self.budget()
        with mock.patch.object(prefix, "prepare_layer_math", return_value=prepared), \
                mock.patch.object(prefix, "solve_selected", return_value=(selected, np.ones(500, dtype=np.float32), 1.)), \
                mock.patch.object(budget, "check", side_effect=[None, None, None, prefix.Stop("deadline")]):
            with self.assertRaisesRegex(prefix.Stop, "deadline"):
                worker.fit_span(bank, span, None, None, None, native, None, store, budget, self.timing())
        begin, rows = store.commit.call_args.args
        self.assertEqual((begin, len(rows)), (128, 2))
        np.testing.assert_array_equal(rows[0][0], native[selected])
        with self.assertRaisesRegex(prefix.Stop, "fewer_than_500"):
            worker.fit_span(bank, span, None, None, None, native[:499], None, store, budget, self.timing())


if __name__ == "__main__":
    unittest.main()
