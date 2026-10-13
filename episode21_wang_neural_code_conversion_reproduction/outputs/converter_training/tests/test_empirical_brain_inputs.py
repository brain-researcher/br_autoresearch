"""Four adapter fixtures; only in coordinated Slurm compute.

Temporary, fabricated897-image/tiny-native-axis role files exercise actual
selective loading and averaging. No real tau payload, fit or scheduler is used.
The already-authenticated image-role helper alone is replaced by fixture IDs.
"""
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import empirical_brain_inputs as inputs


class EmpiricalBrainInputTests(unittest.TestCase):
    def fixture(self, temporary):
        base = Path(temporary)
        prov = base / "outputs/steward_provisioning"
        (prov / "independent_visual_masks").mkdir(parents=True)
        ids = tuple(f"image-{index:04d}" for index in range(897))
        roles = {"tau_train": tuple((index, {"image_id": image}) for index, image in enumerate(ids))}
        alignment = {"fixture": "authenticated_metadata_only"}
        selection = {"alpha_grid": list(inputs.ALPHAS), "n_feat": 0, "dtype": "float32",
            "required_pairing": "separately_image_average_source_and_target_then_exact_join_on_tau_train_image_ID"}
        lock = {"state": "valid_prospective_lock", "job_id": "46424955", "neural_execution_authorized": False,
            "OOD_access_authorized": False, "module_states": {"laion_brain_loss_converters": "runnable_locked"},
            "regular_image_binding": alignment, "resolved_protocol_table": {"brain_loss_contract": {
                "regularization_value_selection_and_intercept": {"selection": selection}}}}
        (base / "outputs/protocol_lock.yaml").write_text(json.dumps(lock))
        subjects, masks, expected, counts, roots = [], [], {}, {}, {}
        for person, width in (("sub-01", 3), ("sub-03", 2)):
            root = prov / "regular_neural_sealed" / person
            root.mkdir(parents=True)
            roots[person] = root
            native = np.array((12, 18, 22)[:width], dtype=np.int64)
            np.save(root / "native_voxel_indices.npy", native)
            digest = hashlib.sha256((root / "native_voxel_indices.npy").read_bytes()).hexdigest()
            grouped = {image: [] for image in ids}
            rows, values = [], []
            # Participants have different counts, trial orders and widths.
            for repetition in range(3):
                order = range(896, -1, -1) if person == "sub-01" else range(897)
                for index in order:
                    if person == "sub-03" and repetition >= 2 + index % 2:
                        continue
                    value = np.arange(width, dtype=np.float32) + np.float32(index + repetition * 4)
                    if person == "sub-01" and index == 0:
                        value[0] = (np.float32(1e8), np.float32(1), np.float32(-1e8))[repetition]
                    grouped[ids[index]].append(value)
                    rows.append(dict(zip(inputs.TRIAL_FIELDS, (str(len(rows)), person, "ses-01", "run-01",
                        str(len(rows)), str(len(rows)), ids[index], "tau_train"))))
                    values.append(value)
            values = np.asarray(values, dtype=np.float32)
            np.save(root / "tau_train_responses.npy", values)
            np.save(root / "tau_train_finite_packed.npy", np.packbits(np.isfinite(values), axis=1, bitorder="little"))
            with (root / "tau_train_trials.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=inputs.TRIAL_FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            expected[person] = np.stack([np.mean(np.stack(grouped[image]), axis=0) for image in ids])
            counts[person] = tuple(len(grouped[image]) for image in ids)
            grid = {"shape": [10, 10, 10]}
            support = {"rows": len(rows), "voxels": width, "dtype": "float32", "nonfinite_elements": 0,
                "repetitions_preserved": True, "finite_mask_bitorder": "little"}
            subjects.append({"participant": person, "roles": {"tau_train": support}, "native_grid": grid,
                "mask_index_sha256": digest, "column_order": "ascending_full_grid_C_order"})
            masks.append({"participant": person, "grid": grid, "union_voxel_count": width,
                "voxel_indices_sha256": digest})
        subjects.extend({"participant": person} for person in inputs.PARTICIPANTS[2:])
        report = {"publication": "complete", "contains_neural_values": False,
                  "OOD_arrays_formed_retained_summarized": False, "subjects": subjects}
        (prov / "regular_neural_structural_report.json").write_text(json.dumps(report))
        (prov / "independent_visual_masks/mask_report.json").write_text(json.dumps({"subjects": masks}))
        return SimpleNamespace(base=base, prov=prov, roles=([], roles, alignment, {}), ids=ids,
                               roots=roots, expected=expected, counts=counts)

    def patches(self, fixture):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(inputs, "BASE", fixture.base))
        stack.enter_context(mock.patch.object(inputs, "PROV", fixture.prov))
        stack.enter_context(mock.patch.object(inputs, "_image_roles", return_value=fixture.roles))
        return stack

    def budget(self, check=lambda: None):
        return SimpleNamespace(allocation={"job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname().split(".")[0], "job_end_epoch": time.time() + 120},
            deadline=time.monotonic() + 60, check=check)

    def bind(self, budget=None, resolver=None):
        return inputs.bind_brain_inputs(object(), require_admitted_context=resolver or (lambda *args, **kw: None),
            source="sub-01", target="sub-03", budget=budget or self.budget())

    def test_independent_full_repetition_means_and_only_tau_train_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            opened, maps, admissions = [], [], []
            actual_load = np.load
            def load(path, **kwargs):
                result = actual_load(path, **kwargs)
                if isinstance(path, Path):
                    opened.append(path.name)
                    self.assertEqual(kwargs["mmap_mode"], "r")
                    maps.append(result)
                return result
            with self.patches(fixture), mock.patch.object(inputs.np, "load", side_effect=load):
                bound = self.bind(resolver=lambda context, **kw: admissions.append(kw))
            self.assertEqual(len(admissions), 4)
            self.assertTrue(all(value["operation"] == "bind_brain_inputs" for value in admissions))
            self.assertEqual(opened, ["tau_train_responses.npy", "tau_train_finite_packed.npy"] * 2)
            self.assertTrue(all(array._mmap.closed for array in maps))
            self.assertEqual(bound.folds.image_ids, fixture.ids)
            for response in (bound.source, bound.target):
                self.assertEqual(response.image_ids, fixture.ids)
                self.assertEqual(response.role, "tau_train")
                self.assertEqual(response.values.dtype, np.float32)
                self.assertFalse(response.values.flags.writeable)
                np.testing.assert_array_equal(response.values, fixture.expected[response.participant])
            self.assertEqual(bound.source.values[0, 0], 0.)
            self.assertEqual(bound.source.native_voxel_ids, (12, 18, 22))
            self.assertEqual(bound.target.native_voxel_ids, (12, 18))
            self.assertEqual(bound.source_repetition_counts, fixture.counts["sub-01"])
            self.assertEqual(bound.target_repetition_counts, fixture.counts["sub-03"])

    def test_wrong_role_or_native_binding_refuses_before_response_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            # Invalid target metadata must prevent even the source payload read.
            path = fixture.roots["sub-03"] / "tau_train_trials.tsv"
            original = path.read_text()
            path.write_text(original.replace("\ttau_train\n", "\ttau_test\n", 1))
            with self.patches(fixture), mock.patch.object(inputs.np, "load") as forbidden:
                with self.assertRaisesRegex(ValueError, "trial_role"):
                    self.bind()
                forbidden.assert_not_called()
            path.write_text(original)
            native = fixture.roots["sub-01"] / "native_voxel_indices.npy"
            np.save(native, np.array([12, 18, 23], dtype=np.int64))
            with self.patches(fixture), mock.patch.object(inputs.np, "load") as forbidden:
                with self.assertRaisesRegex(ValueError, "native_index_source_binding"):
                    self.bind()
                forbidden.assert_not_called()

    def test_admission_precedes_IO_and_cancellation_closes_both_maps(self):
        with mock.patch.object(inputs, "_metadata") as forbidden:
            with self.assertRaisesRegex(PermissionError, "not admitted"):
                self.bind(resolver=mock.Mock(side_effect=PermissionError("not admitted")))
            forbidden.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            maps = []
            actual_load = np.load
            def load(path, **kwargs):
                result = actual_load(path, **kwargs)
                if isinstance(result, np.memmap):
                    maps.append(result)
                return result
            def check():
                if len(maps) == 2:
                    raise TimeoutError("caller deadline")
            with self.patches(fixture), mock.patch.object(inputs.np, "load", side_effect=load):
                with self.assertRaisesRegex(TimeoutError, "caller deadline"):
                    self.bind(budget=self.budget(check))
            self.assertEqual(len(maps), 2)
            self.assertTrue(all(array._mmap.closed for array in maps))

    def test_nonfinite_flags_or_actual_values_stop_without_dropping_rows(self):
        for change in ("flags", "values"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(temporary)
                root = fixture.roots["sub-01"]
                if change == "flags":
                    path = root / "tau_train_finite_packed.npy"
                    value = np.load(path, allow_pickle=False)
                    value[0, 0] = 0
                else:
                    path = root / "tau_train_responses.npy"
                    value = np.load(path, allow_pickle=False)
                    value[0, 0] = np.nan
                np.save(path, value)
                with self.patches(fixture), self.assertRaisesRegex(inputs.BrainLossNotEvaluable, "nonfinite_required"):
                    self.bind()


if __name__ == "__main__":
    unittest.main()
