"""Five exact recovery/assembly checks; coordinator compute allocation only.

Uses artificial operational metadata and a three-row temporary HDF fixture.
No actual decoder, source payload, scheduler or journal is opened.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoder_assembly as assembly
import reviewed_io_recovery as recovery
import test_decoder_assembly as original_fixtures
import test_reviewed_io_recovery as recovery_fixtures


class DecoderIORecoveryAssemblyTests(unittest.TestCase):
    def fixture(self, base=None):
        return recovery_fixtures.fixture(base_directory=str(assembly.BASE if base is None else base))

    def full_catalog(self):
        review, manifest, _, _, _, _ = self.fixture()
        manifest = deepcopy(manifest)
        manifest["retained"] = []
        fixed = {assembly.span_from_record(row).key for item in manifest["work_items"]
                 if item["work_item_id"] in recovery.WORK_IDS for row in item["spans"]}
        remaining = {person: [] for person in assembly.coverage.PARTICIPANTS}
        for span in assembly.coverage.remaining_spans():
            if span.key not in fixed:
                remaining[span.participant].append(span.record())
        for item in manifest["work_items"]:
            if item["work_item_id"] not in recovery.WORK_IDS:
                item.update(spans=[], units_requested=0)
        for index, person in enumerate(assembly.coverage.PARTICIPANTS):
            rows = remaining[person]
            manifest["work_items"][index].update(participant=person, spans=rows,
                units_requested=sum(assembly.span_from_record(row).count for row in rows))
        return review, manifest

    def ref(self, item, *, last_two=False):
        span = assembly.span_from_record(item["committed_spans"][-1])
        if last_two:
            span = replace(span, start=span.stop - 2)
        return assembly.FragmentRef(item["predecessor_fragment"], "grouped_channels",
            f"{span.layer}/channel_{span.channel:03d}", item["predecessor_status"], span)

    def complete_record(self, refs, states, review=None):
        record = {"schema_version": "ep21.assembled_decoder.v1",
            "status": "complete_fixed_decoder_collection", "complete_decoder": True,
            "physical_rows_validated": assembly.coverage.TOTAL, "units": assembly.coverage.TOTAL,
            "participants": list(assembly.coverage.PARTICIPANTS), "layers": list(assembly.coverage.prefix.LAYERS),
            "layer_shapes": dict(zip(assembly.coverage.prefix.LAYERS, map(list, assembly.coverage.prefix.SHAPES))),
            "normalization": assembly.NORMALIZATION, "source_fragment_axes": assembly.AXES,
            "fragments": [ref.record() for ref in refs], "normalization_states": [state.record() for state in states]}
        if review is not None:
            record[recovery.FIELD] = review
        return record

    def test_optional_catalog_retains_full_scope_exact_prefixes_and_complements(self):
        review, manifest = self.full_catalog()
        with mock.patch.object(assembly, "_read_json", side_effect=AssertionError("catalog is metadata-only")):
            ordinary, states = assembly.planned_catalog(manifest, ())
            unchanged, _ = assembly.planned_catalog(manifest, (), recovery_review=None)
            recovered, other_states = assembly.planned_catalog(manifest, (), recovery_review=review)
        self.assertEqual(ordinary, unchanged)
        self.assertEqual(states, other_states)
        self.assertEqual(assembly.verify_catalog(recovered, states)["units"], 74306440)
        assembly._validate_recovery_catalog(recovered, review)
        old_units = new_units = 0
        for item in review["work_items"]:
            old = [ref for ref in recovered if ref.path == item["predecessor_fragment"]]
            new = [ref for ref in recovered if ref.path == item["output_directory"] + "/channels.h5"]
            self.assertEqual([ref.span.record() for ref in old], item["committed_spans"])
            self.assertEqual([ref.span.record() for ref in new], item["spans"])
            old_units += sum(ref.span.count for ref in old)
            new_units += sum(ref.span.count for ref in new)
        self.assertEqual((old_units, new_units), (3320576, 1697024))
        for delta in (-1, 1):
            malformed = deepcopy(review)
            malformed["work_items"][0]["spans"][0]["spatial_Fortran_interval"][0] += delta
            with self.assertRaises(ValueError):
                assembly.planned_catalog(manifest, (), recovery_review=malformed)

    def test_exception_applies_only_after_ordinary_original_identity_and_prefix_checks(self):
        review, _, _, closed, _, _ = self.fixture()
        item = review["work_items"][0]
        ref, record = self.ref(item), closed[item["work_item_id"]]
        state = assembly.required_state(item["participant"])
        before = deepcopy(record)
        with self.assertRaisesRegex(ValueError, "uncertain marker"):
            assembly.validate_terminal_record(ref, record, state)
        assembly.validate_terminal_record(ref, record, state, recovery_review=review)
        self.assertEqual(record, before)
        for update in ({"training_role": "tau_train"}, {"tau_train_values_read": True},
                       {"participant": "sub-07"}, {"shared_state_reference": "wrong-state"},
                       {"committed_spans": []}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                assembly.validate_terminal_record(ref, {**record, **update}, state, recovery_review=review)
        with self.assertRaisesRegex(ValueError, "exact committed coverage"):
            assembly.validate_terminal_record(replace(ref, span=replace(ref.span, stop=ref.span.stop + 1)),
                                              record, state, recovery_review=review)
        unrelated = replace(ref, path=assembly.FULL_ROOT + "/shards/work_0048/channels.h5",
                            status_path=assembly.FULL_ROOT + "/shards/work_0048/work_item_status.json")
        with self.assertRaisesRegex(ValueError, "uncertain marker"):
            assembly.validate_terminal_record(unrelated, record, state, recovery_review=review)

    def test_sparse_scan_preserves_weights_bias_and_excludes_marker_zero_tail(self):
        helper = original_fixtures.DecoderAssemblyTests()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(assembly, "BASE", Path(temporary)):
            review, _, _, closed, _, _ = self.fixture(temporary)
            item = review["work_items"][0]
            ref = self.ref(item, last_two=True)
            physical = replace(ref, span=replace(ref.span, stop=ref.span.stop + 1))
            path = Path(temporary) / ref.path
            path.parent.mkdir(parents=True)
            handle, group = helper.fragment(path, physical)
            group["state"][2] = 0
            group["weights"][2] = np.nan
            group["bias"][:] = [9., 10., np.nan]
            expected = group["weights"][:2].copy()
            handle.close()
            (Path(temporary) / ref.status_path).write_text(json.dumps(closed[item["work_item_id"]]))
            arrays, binding = helper.arrays()
            binding["participant"] = item["participant"]
            normalization = assembly.validate_normalization(arrays, binding, item["participant"])
            state = assembly.required_state(item["participant"])
            with self.assertRaisesRegex(ValueError, "uncertain marker"):
                list(assembly._scan_fragments(temporary, [ref], state, normalization, lambda: None, 2))
            blocks = list(assembly._scan_fragments(temporary, [ref], state, normalization, lambda: None, 2,
                                                  recovery_review=review))
            self.assertEqual(len(blocks), 1)
            np.testing.assert_array_equal(blocks[0].weights, expected)
            np.testing.assert_array_equal(blocks[0].bias, [9., 10.])
            with self.assertRaisesRegex(ValueError, "exact committed coverage"):
                list(assembly._scan_fragments(temporary, [physical], state, normalization, lambda: None, 2,
                                              recovery_review=review))
            with h5py.File(path, "r+") as handle:
                handle[ref.group]["weights"][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "invalid fitted coefficient"):
                list(assembly._scan_fragments(temporary, [ref], state, normalization, lambda: None, 2,
                                              recovery_review=review))

    def test_manifest_binding_survives_load_and_sparse_reader_and_cannot_be_dropped(self):
        review, source = self.full_catalog()
        refs, states = assembly.planned_catalog(source, (), recovery_review=review)
        record = self.complete_record(refs, states, review)
        manifest = assembly.validate_complete_manifest(record)
        self.assertEqual(manifest.recovery_review, review)
        missing = dict(record)
        del missing[recovery.FIELD]
        with self.assertRaisesRegex(ValueError, "explicit review binding"):
            assembly.validate_complete_manifest(missing)
        changed = deepcopy(record)
        index = next(index for index, ref in enumerate(refs) if ref.path == review["work_items"][0]["predecessor_fragment"])
        changed["fragments"][index]["path"] = assembly.FULL_ROOT + "/shards/unreviewed/channels.h5"
        with self.assertRaisesRegex(ValueError, "exact reviewed prefixes"):
            assembly.validate_complete_manifest(changed)
        budget = mock.Mock()
        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "_read_json", return_value=record), \
                mock.patch.object(assembly, "_revalidate_recovery_review") as rebind:
            loaded = assembly.load_complete_manifest(assembly.BASE / assembly.MANIFEST, budget=budget)
            rebind.assert_called_once_with(review, budget.check)
        events = []
        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "_revalidate_recovery_review", side_effect=lambda *args: events.append("review")), \
                mock.patch.object(assembly, "_admit_normalization", side_effect=lambda *args: events.append("normalization")), \
                mock.patch.object(assembly, "_scan_fragments", return_value=iter(())) as scan:
            list(assembly.iter_sparse_blocks(loaded, participant="sub-05", layer="conv1_1", channel=35, budget=budget))
            self.assertEqual(events, ["review", "normalization"])
            self.assertEqual(scan.call_args.kwargs, {"recovery_review": review})
        legacy_refs, _ = assembly.planned_catalog(source, ())
        legacy = assembly.validate_complete_manifest(self.complete_record(legacy_refs, states))
        self.assertIsNone(legacy.recovery_review)

    def test_admission_requires_registered_review_closed_fresh_jobs_and_unchanged_binding(self):
        review, manifest, jobs, closed, reports, quarantine_review = self.fixture()
        account = {"jobs": deepcopy(jobs), "batches": {}}
        documents = {assembly.BASE / recovery.REVIEW_RELATIVE: review,
                     assembly.BASE / "outputs/decoder_training_full_authorization.json": {},
                     assembly.BASE / assembly.FULL_ROOT / "work_manifest.json": manifest}
        documents.update({assembly.BASE / path: report for path, report in reports.items()})
        for index, item in enumerate(review["work_items"]):
            work, batch_id = item["work_item_id"], "recovery_" + item["work_item_id"]
            documents[assembly.BASE / item["predecessor_status"]] = closed[work]
            record, job = recovery_fixtures.completed_record(review, work, raw=str(9001 + index),
                                                             base_directory=str(assembly.BASE))
            job = dict(job, batch_id=batch_id, job_id=f"900_{item['array_index']}")
            account["jobs"][job["job_id"]] = job
            account["batches"][batch_id] = {
                **recovery_fixtures.batch(review, quarantine_review, (work,)), "accepted_id": "900"}
            documents[assembly.BASE / item["output_directory"] / "work_item_status.json"] = record
        account["batches"]["assembly"] = {"accepted_id": "10000", recovery.FIELD: review}
        account["jobs"]["10000"] = {"job_id": "10000", "phase": "assembly", "logical_id": "assembly",
                                     "batch_id": "assembly", "terminal": None}
        def admit(current=account, current_documents=documents):
            with mock.patch.object(assembly, "_read_json", side_effect=lambda path: deepcopy(current_documents[path])), \
                    mock.patch.object(assembly, "_recovery_account", return_value=current):
                return assembly._admit_recovery_review(assembly.BASE / recovery.REVIEW_RELATIVE,
                    {}, manifest, {"job_id": "10000"}, lambda: None)
        self.assertEqual(admit(), review)
        bad = deepcopy(account)
        del bad["batches"]["assembly"][recovery.FIELD]
        with self.assertRaisesRegex(ValueError, "explicitly register"):
            admit(bad)
        bad = deepcopy(account)
        first_fresh = f"900_{review['work_items'][0]['array_index']}"
        bad["jobs"][first_fresh]["terminal"]["state"] = "FAILED"
        with self.assertRaisesRegex(ValueError, "complete_charged"):
            admit(bad)
        with mock.patch.object(assembly, "_read_json", return_value={}), \
                mock.patch.object(assembly, "_recovery_account") as forbidden:
            with self.assertRaisesRegex(ValueError, "differs from the fixed review"):
                assembly._revalidate_recovery_review(review, lambda: None)
            forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
