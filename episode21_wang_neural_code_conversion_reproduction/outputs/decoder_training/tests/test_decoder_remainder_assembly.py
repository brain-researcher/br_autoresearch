"""Exact remainder catalog and sparse-reader regressions; compute allocation only.

These fixtures are orchestration examples, not authenticated empirical closeouts.
The shared remainder helper separately tests registration and charge admission.
No scientific payload is read; only tiny temporary coefficient fixtures are made.
"""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoder_assembly as assembly
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainders
import test_decoder_assembly as original_fixtures
import test_decoder_deadline_remainders as remainder_fixtures


class DecoderRemainderAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = {("sub-01", "conv1_2", channel): f"work_{index:04d}"
                    for index, channel in zip((5, 6, 7), (30, 31, 32))}
        originals, others = {}, {person: [] for person in assembly.coverage.PARTICIPANTS}
        for span in assembly.coverage.remaining_spans():
            if span.key in selected:
                identifier = selected[span.key]
                originals[identifier] = {"work_item_id": identifier, "participant": span.participant,
                    "spans": [span.record()], "units_requested": span.count}
            else:
                others[span.participant].append(span.record())
        cls.manifest = {"retained": [], "work_items": [originals[key] for key in sorted(originals)] + [
            {"work_item_id": f"fixture_{person}", "participant": person, "spans": rows,
             "units_requested": sum(row["spatial_Fortran_interval"][1]
                                    - row["spatial_Fortran_interval"][0] for row in rows)}
            for person, rows in others.items()]}
        cls.remainders = {}
        for number, identifier in enumerate(sorted(originals), start=5):
            row = deepcopy(originals[identifier]["spans"][0])
            committed = deepcopy(row)
            committed["spatial_Fortran_interval"][1] = 1234
            row["spatial_Fortran_interval"][0] = 1234
            cls.remainders[identifier] = {"work_item_id": identifier, "array_index": number - 1,
                "participant": "sub-01", "predecessor_job_id": f"9000_{number - 1}",
                "predecessor_allocation_job_id": str(9100 + number),
                "predecessor_status": assembly.FULL_ROOT + f"/shards/{identifier}/work_item_status.json",
                "output_directory": assembly.FULL_ROOT + f"/remainder_shards_001/{identifier}",
                "spans": [row], "committed_spans": [committed],
                "units_requested": row["spatial_Fortran_interval"][1] - 1234, "resources": {}}

    def catalog(self, replacements):
        return assembly.planned_catalog(self.manifest, (), remainder_items=replacements)

    def test_default_catalog_keeps_original_paths_and_does_not_discover_remainders(self):
        with mock.patch.object(assembly, "_read_json", side_effect=AssertionError("metadata-only catalog")):
            ordinary, states = assembly.planned_catalog(self.manifest, ())
            explicit_none, other_states = self.catalog(None)
        self.assertEqual(ordinary, explicit_none)
        self.assertEqual(states, other_states)
        self.assertFalse(any("remainder_shards" in ref.path for ref in ordinary))
        self.assertEqual(sum(ref.span.count for ref in ordinary), assembly.coverage.TOTAL)

    def test_three_replacements_preserve_exact_target_and_other_physical_identities(self):
        ordinary, states = assembly.planned_catalog(self.manifest, ())
        original_record = deepcopy(self.remainders)
        replaced, new_states = self.catalog(self.remainders)
        self.assertEqual(self.remainders, original_record)
        self.assertEqual(states, new_states)
        self.assertEqual(assembly.verify_catalog(replaced, states)["units"], 74_306_440)
        affected_paths = set()
        for identifier, item in self.remainders.items():
            old_path = assembly.FULL_ROOT + f"/shards/{identifier}/channels.h5"
            new_path = item["output_directory"] + "/channels.h5"
            affected_paths.update((old_path, new_path))
            old = [ref for ref in replaced if ref.path == old_path]
            new = [ref for ref in replaced if ref.path == new_path]
            self.assertEqual([ref.span.record() for ref in old], item["committed_spans"])
            self.assertEqual([ref.span.record() for ref in new], item["spans"])
            self.assertTrue(all(ref.status_path == item["predecessor_status"] for ref in old))
            self.assertTrue(all(ref.status_path == item["output_directory"] + "/work_item_status.json"
                                for ref in new))
        self.assertEqual([ref for ref in ordinary if ref.path not in affected_paths],
                         [ref for ref in replaced if ref.path not in affected_paths])

    def test_each_assignment_rejects_gaps_overlaps_duplicates_and_exchanged_scopes(self):
        cases = []
        for delta in (-1, 1):
            bad = deepcopy(self.remainders)
            bad["work_0005"]["spans"][0]["spatial_Fortran_interval"][0] += delta
            bad["work_0005"]["units_requested"] -= delta
            cases.append(bad)
        bad = deepcopy(self.remainders)
        bad["work_0005"]["committed_spans"] *= 2
        cases.append(bad)
        # Exchanging complete channel scopes preserves global coverage but must
        # fail the binding to each original logical assignment.
        bad = deepcopy(self.remainders)
        for field in ("committed_spans", "spans"):
            bad["work_0005"][field], bad["work_0006"][field] = (
                bad["work_0006"][field], bad["work_0005"][field])
        cases.append(bad)
        for index, value in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((ValueError, RuntimeError)):
                self.catalog(value)

    def test_optional_catalog_requires_exact_three_attempt_paths_and_counts(self):
        cases = []
        for field, value in (("predecessor_status", self.remainders["work_0006"]["predecessor_status"]),
                             ("output_directory", self.remainders["work_0006"]["output_directory"]),
                             ("work_item_id", "work_0006"), ("participant", "sub-03"),
                             ("units_requested", 1)):
            bad = deepcopy(self.remainders)
            bad["work_0005"][field] = value
            cases.append(bad)
        bad = deepcopy(self.remainders)
        del bad["work_0007"]
        cases.append(bad)
        bad = deepcopy(self.remainders)
        bad["work_0008"] = deepcopy(bad["work_0007"])
        cases.append(bad)
        for index, value in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((ValueError, RuntimeError)):
                self.catalog(value)

    def test_sparse_scan_joins_two_origins_without_reading_old_uncommitted_tail(self):
        fixture = original_fixtures.DecoderAssemblyTests(
            "test_grouped_scan_opens_one_file_and_status_once_for_multiple_channels")
        physical_old = fixture.ref()
        identifier = "work_0005"
        old_directory = assembly.FULL_ROOT + f"/shards/{identifier}"
        new_directory = assembly.FULL_ROOT + f"/remainder_shards_001/{identifier}"
        physical_old = replace(physical_old, path=old_directory + "/channels.h5",
                               status_path=old_directory + "/work_item_status.json")
        committed = replace(physical_old, span=replace(physical_old.span, stop=194))
        remainder = replace(physical_old, path=new_directory + "/channels.h5",
            status_path=new_directory + "/work_item_status.json",
            span=replace(physical_old.span, start=194))
        state, normalization = assembly.required_state("sub-01"), fixture.normalization()
        with tempfile.TemporaryDirectory() as temporary:
            old_path, new_path = Path(temporary) / physical_old.path, Path(temporary) / remainder.path
            old_path.parent.mkdir(parents=True)
            new_path.parent.mkdir(parents=True)
            handle, group = fixture.fragment(old_path, physical_old)
            group["state"][1:] = 0
            group["weights"][1:] = np.nan
            group["bias"][0] = 9
            handle.close()
            handle, group = fixture.fragment(new_path, remainder)
            group["bias"][:] = [10, 11]
            handle.close()
            for ref, status in ((committed, "partial_work_item_stopped"),
                                (remainder, "complete_work_item_only")):
                record = {"schema_version": "ep21.decoder_work_item.v1", "status": status,
                    "participant": "sub-01", "training_role": "subject_unique", "training_images": 4712,
                    "predictions_or_scores_computed": False, "tau_train_values_read": False,
                    "tau_test_values_read": False, "OOD_access": False,
                    "shared_state_reference": str(assembly.BASE / state.path),
                    "committed_spans": [ref.span.record()]}
                (Path(temporary) / ref.status_path).write_text(json.dumps(record))
            blocks = list(assembly._scan_fragments(temporary, [committed, remainder], state,
                                                   normalization, lambda: None, 1))
            observed = sorted((int(block.full_layer_Fortran_index[0]), float(block.bias[0]))
                              for block in blocks)
            expected = [(assembly.coverage.coordinates(committed.span.shape, 511, index)[1], float(value))
                        for index, value in ((193, 9), (194, 10), (195, 11))]
            self.assertEqual(observed, expected)
            with self.assertRaisesRegex(ValueError, "exact committed coverage"):
                list(assembly._scan_fragments(temporary, [physical_old], state,
                                              normalization, lambda: None, 1))

    def test_metadata_admission_requires_exact_registered_success_and_preserved_predecessor_charge(self):
        authority, manifest, plan, review, predecessors, account = remainder_fixtures.fixture()
        account["batches"]["remainders"] = {"accepted_id": "900", remainders.FIELD: plan}
        account["batches"]["assembly"] = {"accepted_id": "1000", remainders.FIELD: plan}
        account["jobs"]["1000"] = {"job_id": "1000", "phase": "assembly", "logical_id": "assembly",
                                    "batch_id": "assembly", "terminal": None}
        documents = {assembly.BASE / remainders.PLAN_RELATIVE: plan,
                     assembly.BASE / remainders.REVIEW_RELATIVE: review}
        for item in plan["work_items"]:
            work, index = item["work_item_id"], item["array_index"]
            documents[assembly.BASE / item["predecessor_status"]] = predecessors[work]
            raw_job = str(9000 + index)
            account["jobs"][f"900_{index}"] = {"job_id": f"900_{index}", "phase": "workers",
                "logical_id": work, "batch_id": "remainders", "reserved_CPU_seconds": 28800,
                "terminal": {"state": "COMPLETED", "exit_code": "0:0", "allocated_cpus": 1,
                             "allocation_job_id": raw_job, "elapsed_seconds": 60}}
            closed = deepcopy(predecessors[work])
            closed.update(status="complete_work_item_only", resources=item["resources"],
                requested_spans=item["spans"], committed_spans=item["spans"], remaining_spans=[],
                units_requested=item["units_requested"], completed_units=item["units_requested"],
                remainder_plan=str(assembly.BASE / remainders.PLAN_RELATIVE),
                predecessor_job_id=item["predecessor_job_id"], predecessor_status=item["predecessor_status"])
            closed.pop("error")
            closed.pop("error_type")
            closed["allocation"]["job_id"] = raw_job
            documents[assembly.BASE / item["output_directory"] / "work_item_status.json"] = closed

        def admit(current_account, current_documents, path=None):
            with mock.patch.object(accounting, "locked_events", return_value=nullcontext(([], None))), \
                    mock.patch.object(accounting, "replay", return_value=current_account), \
                    mock.patch.object(assembly, "_read_json", side_effect=lambda name: deepcopy(current_documents[name])):
                return assembly._admit_remainder_items(
                    path or assembly.BASE / remainders.PLAN_RELATIVE, authority, manifest,
                    {"job_id": "1000"}, lambda: None)

        admitted = admit(account, documents)
        self.assertEqual(tuple(admitted), remainders.WORK_IDS)
        self.assertEqual(sum(item["units_requested"] for item in admitted.values()), 355840)
        changed = deepcopy(account)
        del changed["batches"]["assembly"][remainders.FIELD]
        with self.assertRaisesRegex(ValueError, "explicitly register"):
            admit(changed, documents)
        changed = deepcopy(account)
        changed["jobs"]["46839670_4"]["terminal"]["elapsed_seconds"] -= 1
        with self.assertRaisesRegex(ValueError, "terminal_charge"):
            admit(changed, documents)
        changed = deepcopy(account)
        changed["jobs"]["900_4"]["terminal"]["state"] = "FAILED"
        with self.assertRaisesRegex(ValueError, "successfully_charged"):
            admit(changed, documents)
        changed = deepcopy(documents)
        output = assembly.BASE / plan["work_items"][0]["output_directory"] / "work_item_status.json"
        changed[output]["committed_spans"] = predecessors["work_0005"]["requested_spans"]
        with self.assertRaisesRegex(ValueError, "exact_fresh_spans"):
            admit(account, changed)
        with self.assertRaisesRegex(ValueError, "fixed explicit remainder"):
            admit(account, documents, assembly.BASE / "outputs/decoder_training/other.json")


if __name__ == "__main__":
    unittest.main()
