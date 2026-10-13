"""Closed-file storage accounting fixtures; run only in coordinated Slurm.

No HDF5, scientific input or scheduler is opened. Fake retained coverage in the
aggregate test isolates resource admission from previously checked readers.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_execution_contract as contract
import plan_full_decoder_work as planner


class ClosedLayerStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        patch = mock.patch.object(contract, "BASE", self.base)
        patch.start()
        self.addCleanup(patch.stop)
        self.root = self.base / "outputs/decoder_training"
        self.layer = self.root / "layer_continuation_001"
        self.layer.mkdir(parents=True)
        (self.layer / "channels.h5").write_bytes(b"closed fragment fixture")
        (self.layer / "requested_units.json").write_text("{}\n")
        for suffix in ("out", "err"):
            (self.root / f"continue_layer001-99.{suffix}").write_text("closed\n")
        self.included = {"job_id": "99", "reserved_CPU_seconds": 86400,
                         "durable_bytes": 25_000_000_000}
        self.report = {
            "allocation": {"job_id": "99"}, "resources": {"durable_bytes": 25_000_000_000},
            "status": "partial_layer_continuation_stopped", "closeout_errors": [],
            "completed_units_is_lower_bound": False,
            "remaining_interval_status": "exact_uncommitted_coverage",
        }
        self.retained = [{"kind": "layer_continuation",
                          "status": str((self.layer / "continuation_status.json").relative_to(self.base)),
                          "closed_parent": {"job_id": "99", "state": "CANCELLED"}}]
        self.authority = {}
        self.bind()

    def bind(self):
        names = ["channels.h5", "requested_units.json"]
        if (self.layer / "complete_layer_manifest.json").exists():
            names.append("complete_layer_manifest.json")
        self.report["durable_bytes_before_final_status"] = sum((self.layer / n).stat().st_size for n in names)
        (self.layer / "continuation_status.json").write_text(json.dumps(self.report))
        names.append("continuation_status.json")
        files = {str((self.layer / n).relative_to(self.base)): (self.layer / n).stat().st_size for n in names}
        self.authority["closed_included_layer_storage"] = {
            "job_id": "99", "files": files,
            "metadata_allowance_bytes": contract.CLOSED_LAYER_METADATA_BYTES,
            "charged_bytes": sum(files.values()) + contract.CLOSED_LAYER_METADATA_BYTES,
        }

    def charge(self):
        return contract.included_layer_storage_charge(self.authority, self.included, self.retained)

    def test_default_reservation_and_explicit_closed_charge_preserve_budget_identity(self):
        self.assertEqual(contract.included_layer_storage_charge({}, self.included, []), 25_000_000_000)
        self.assertEqual(contract.included_layer_storage_charge({}, None, []), 0)
        authority = {"cpu_budget_mode": accounting.MODE, "included_layer_job": self.included,
                     "resources": {"maximum_allocated_CPU_seconds": 1_800_000,
                                   "durable_bytes": 500_000_000_000}}
        before = accounting.budget_definition(authority)
        authority.update(self.authority)
        self.assertEqual(before, accounting.budget_definition(authority))
        self.assertEqual(self.charge(), self.authority["closed_included_layer_storage"]["charged_bytes"])
        rows = [{"event": "budget_opened", "budget": before},
                {"event": "terminal_charge", "job_id": "99", "allocation_job_id": "99",
                 "allocated_cpus": 1, "elapsed_seconds": 17190, "state": "CANCELLED",
                 "exit_code": "0:15", "observed_utc": "2026-10-06T00:00:00Z", "evidence": "fixture"}]
        replay = accounting.replay(rows, authority)
        self.assertEqual((replay["actual_CPU_seconds"], replay["outstanding_reserved_CPU_seconds"]), (17190, 0))

    def test_live_wrong_or_absent_included_job_cannot_reclaim(self):
        with self.assertRaises(contract.Stop):
            contract.included_layer_storage_charge(self.authority, None, self.retained)
        for key, value in (("state", "RUNNING"), ("job_id", "100")):
            with self.subTest(key=key):
                rows = deepcopy(self.retained)
                rows[0]["closed_parent"][key] = value
                with self.assertRaisesRegex(contract.Stop, "closed_included_parent"):
                    contract.included_layer_storage_charge(self.authority, self.included, rows)

    def test_growth_missing_inventory_and_symlink_fail_closed(self):
        with (self.layer / "channels.h5").open("ab") as stream:
            stream.write(b"growth")
        with self.assertRaisesRegex(contract.Stop, "size_changed"):
            self.charge()
        self.bind()
        self.authority["closed_included_layer_storage"]["files"].pop(self.retained[0]["status"])
        with self.assertRaisesRegex(contract.Stop, "all_named_closed_layer_files"):
            self.charge()
        self.bind()
        path = self.layer / "channels.h5"
        moved = self.layer / "other"
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaisesRegex(contract.Stop, "regular_file"):
            self.charge()

    def test_terminal_uncertainty_or_charge_tampering_cannot_reduce_storage(self):
        for field, value in (("completed_units_is_lower_bound", True),
                             ("closeout_errors", [{"affects_fragment": False}]),
                             ("status", "admitted_pending")):
            with self.subTest(field=field):
                original = deepcopy(self.report)
                self.report[field] = value
                self.bind()
                with self.assertRaisesRegex(contract.Stop, "terminal_accounting"):
                    self.charge()
                self.report = original
        self.bind()
        self.authority["closed_included_layer_storage"]["charged_bytes"] -= 1
        with self.assertRaisesRegex(contract.Stop, "charge_must_cover"):
            self.charge()

    def test_optional_manifest_must_be_included_and_logs_fit_reserve(self):
        (self.layer / "complete_layer_manifest.json").write_text("{}\n")
        with self.assertRaisesRegex(contract.Stop, "all_named_closed_layer_files"):
            self.charge()
        self.bind()
        self.charge()
        with (self.root / "continue_layer001-99.out").open("wb") as stream:
            stream.truncate(contract.CLOSED_LAYER_METADATA_BYTES + 1)
        with self.assertRaisesRegex(contract.Stop, "metadata_allowance"):
            self.charge()

    def test_full_coverage_fits_without_changing_500GB_ceiling(self):
        spans = list(contract.coverage.retained_spans())
        spans.extend(contract.coverage.Span("sub-01", "conv1_1", c, 0, 50176) for c in range(1, 14))
        spans.append(contract.coverage.Span("sub-01", "conv1_1", 14, 0, 12544))
        plan = planner.plan_from_spans(spans)
        plan["retained"] = self.retained
        authority = {**self.authority, "cpu_budget_mode": accounting.MODE,
                     "included_layer_job": self.included, "work_manifest": contract.MANIFEST_RELATIVE,
                     "reuse_completed_units": 716160,
                     "state_build_resources": plan["state_build_resources"],
                     "assembly_resources": plan["assembly_resources"],
                     "resources": {"worker_jobs": len(plan["work_items"]), "state_build_jobs": 1,
                                   "assembly_jobs": 1, "maximum_concurrent_jobs": 8,
                                   "maximum_allocated_CPU_seconds": 1_800_000,
                                   "durable_bytes": 500_000_000_000, "peak_transient_bytes": 16_000_000_000}}
        with mock.patch.object(contract, "read_retained_coverage", return_value=spans), \
                mock.patch.object(contract, "validate_retained_continuation_causes"):
            without_reclaim = deepcopy(authority)
            del without_reclaim["closed_included_layer_storage"]
            with self.assertRaisesRegex(contract.Stop, "aggregate_full_stage_caps"):
                contract.validate_work_manifest(plan, without_reclaim)
            admitted = contract.validate_work_manifest(plan, authority)
            self.assertEqual(len(admitted), 75)
            total = plan["prospective_aggregate"]["maximum_new_durable_bytes"] + self.charge() + accounting.JOURNAL_BYTES
            authority["resources"]["durable_bytes"] = total - 1
            with self.assertRaisesRegex(contract.Stop, "aggregate_full_stage_caps"):
                contract.validate_work_manifest(plan, authority)


if __name__ == "__main__":
    unittest.main()
