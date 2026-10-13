"""Three source-free interface checks; coordinator compute only, UNRUN.

The complete 224-image/five-owner identities are retained, with explicitly
small fixture native axes. Common locked-truth admission and the existing
truth reader's numerical checks are reused, not repeated here.
"""
from contextlib import contextmanager, ExitStack
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_full_empirical_inputs as inputs


IDS = tuple(f"image-{number:03d}" for number in range(224))
IDENTITY = {"role": "tau_test", "dtype": "float32", "extraction_status": "complete",
            "checkpoint": "fixture-checkpoint", "preprocessing": "fixture-preprocessing"}


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=inputs.first.TRIAL_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def environment():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        base = Path(directory).resolve()
        prov = base / "outputs/steward_provisioning"
        truth_root = prov / "regular_vgg19_sealed/tau_test"
        truth_root.mkdir(parents=True)
        widths = {owner: index + 3 for index, owner in enumerate(inputs.PARTICIPANTS)}
        neural = {"publication": "complete", "contains_neural_values": False,
                  "OOD_arrays_formed_retained_summarized": False, "subjects": []}
        masks, sources, rows_by_owner, values_by_owner, expected = {"subjects": []}, {}, {}, {}, {}
        for number, owner in enumerate(inputs.PARTICIPANTS):
            root, width = prov / "regular_neural_sealed" / owner, widths[owner]
            root.mkdir(parents=True)
            # Stored repetition order differs from the locked common image order;
            # owners have different repetition counts for the first image.
            order = list(range(223, 0, -1)) + [0] * (3 + number)
            native = np.arange(width, dtype=np.int64) * 3 + number + 1
            values = np.asarray([[number * 1000 + image * 3 + column for column in range(width)]
                                 for image in order], dtype=np.float32)
            values[223:226] = np.array([1e8, 1, -1e8], dtype=np.float32)[:, None]
            values[226:] = 2
            # A constant native coordinate must not be dropped by this adapter.
            values[:, -1] = 5
            oracle = np.asarray([[number * 1000 + image * 3 + column for column in range(width)]
                                 for image in range(224)], dtype=np.float32)
            oracle[0] = np.float32(2 * number) / np.float32(3 + number)
            oracle[:, -1] = 5
            np.save(root / "native_voxel_indices.npy", native)
            np.save(root / "tau_test_responses.npy", values)
            np.save(root / "tau_test_finite_packed.npy",
                    np.packbits(np.ones(values.shape, dtype=np.uint8), axis=1, bitorder="little"))
            digest = hashlib.sha256((root / "native_voxel_indices.npy").read_bytes()).hexdigest()
            rows = [{"output_row_index": str(index), "participant": owner, "session": "session-a",
                     "run": "run-a", "beta_index": str(index), "beta_volume_index": str(index),
                     "image_id": IDS[image], "role": "tau_test"} for index, image in enumerate(order)]
            table(root / "tau_test_trials.tsv", rows)
            groups = tuple(tuple(index for index, image in enumerate(order) if image == number)
                           for number in range(224))
            source = inputs.first.SourceMetadata(root, len(order), width, digest, (64,), groups, IDS)
            sources[owner], rows_by_owner[owner], values_by_owner[owner] = source, rows, values
            expected[owner] = oracle
            neural["subjects"].append({"participant": owner,
                "roles": {"tau_test": {"rows": len(order), "voxels": width, "dtype": "float32",
                "repetitions_preserved": True, "finite_mask_bitorder": "little", "nonfinite_elements": 0}},
                "column_order": "ascending_full_grid_C_order", "mask_index_sha256": digest,
                "native_grid": {"shape": [64]}})
            masks["subjects"].append({"participant": owner, "union_voxel_count": width,
                "voxel_indices_sha256": digest, "grid": {"shape": [64]}})
        neural_path = prov / "regular_neural_structural_report.json"
        neural_path.write_text(json.dumps(neural))
        mask_path = prov / "independent_visual_masks/mask_report.json"
        mask_path.parent.mkdir()
        mask_path.write_text(json.dumps(masks))
        truth = inputs.first.TruthMetadata(truth_root, IDS, IDENTITY)
        # No old module globals/readers are patched. The common helper's existing
        # sub03/truth metadata validation is outside this new owner composition.
        for name, value in (("BASE", base), ("PROV", prov), ("NATIVE_VOXELS", widths)):
            stack.enter_context(mock.patch.object(inputs, name, value))
        stack.enter_context(mock.patch.object(inputs, "_common_metadata", return_value=(sources["sub-03"], truth)))
        stack.enter_context(mock.patch.object(inputs, "_require_reader_context", side_effect=lambda budget: budget.check()))
        budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"}, check=lambda: None)
        context, admitted, readers, maps, paths = object(), [], [], [], []

        def resolver(actual, **kwargs):
            if (actual is not context or kwargs["budget"] is not budget or kwargs["target"] is not None
                    or kwargs["execution_options"] is not None):
                raise AssertionError("unexpected external admission identity")
            operation, owner = kwargs["operation"], kwargs["source"]
            if ((operation == inputs.READ_OWNER and owner not in inputs.PARTICIPANTS)
                    or (operation != inputs.READ_OWNER and owner is not None)):
                raise AssertionError("wrong fixed owner/operation binding")
            admitted.append((operation, owner))

        actual_load, actual_reader = inputs._load_array, inputs.TruthReader

        def load(path, **kwargs):
            if isinstance(path, Path) and path.name == "tau_test_responses.npy":
                if not all(array._mmap.closed for array in maps):
                    raise AssertionError("another owner's maps survived into the next owner")
            result = actual_load(path, **kwargs)
            if isinstance(result, np.memmap):
                if kwargs.get("mmap_mode") != "r":
                    raise AssertionError("neural maps must be read-only")
                maps.append(result)
                paths.append(Path(path))
            return result

        def reader(*args, **kwargs):
            result = actual_reader(*args, **kwargs)
            readers.append(result)
            return result

        stack.enter_context(mock.patch.object(inputs, "_load_array", side_effect=load))
        stack.enter_context(mock.patch.object(inputs, "TruthReader", side_effect=reader))
        kwargs = dict(execution_context=context, require_admitted_context=resolver, budget=budget)
        yield SimpleNamespace(base=base, prov=prov, sources=sources, truth=truth, rows=rows_by_owner,
            values=values_by_owner, expected=expected, neural=neural, neural_path=neural_path,
            masks=masks, mask_path=mask_path, kwargs=kwargs, budget=budget, admitted=admitted,
            readers=readers, maps=maps, paths=paths)


class FullFindingAEmpiricalInputsTests(unittest.TestCase):
    def test_exact_five_owner_averaging_native_identity_and_single_truth_reader(self):
        self.assertEqual(inputs.PARTICIPANTS, ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07"))
        self.assertEqual(dict(inputs.NATIVE_VOXELS), {"sub-01": 12295, "sub-03": 12897,
                         "sub-05": 10891, "sub-06": 12023, "sub-07": 12057})
        self.assertEqual(inputs.TEST_IMAGES, 224)
        self.assertEqual(inputs.first.MAX_TRUTH_BANK_BYTES, 67108864)
        with environment() as env:
            bound = inputs.bind_full_finding_a_inputs(**env.kwargs)
            try:
                self.assertEqual(bound.participants, inputs.PARTICIPANTS)
                self.assertEqual(bound.image_ids, IDS)
                self.assertEqual(tuple(bound.responses_by_participant), inputs.PARTICIPANTS)
                self.assertEqual(len(env.readers), 1)
                self.assertIs(bound.truth_reader, env.readers[0])
                self.assertIsNone(bound.truth_reader._features)
                self.assertIsNone(bound.truth_reader._flags)
                for number, owner in enumerate(inputs.PARTICIPANTS):
                    actual = bound.responses_by_participant[owner]
                    self.assertEqual(actual.participant, owner)
                    self.assertEqual((actual.role, actual.aggregation),
                        ("tau_test", "separately_image_averaged_after_role_assignment"))
                    self.assertEqual(actual.image_ids, IDS)
                    self.assertEqual(actual.native_voxel_ids, tuple(3 * column + number + 1
                                     for column in range(number + 3)))
                    np.testing.assert_array_equal(actual.values, env.expected[owner])
                    self.assertEqual(actual.values.dtype, np.float32)
                    self.assertFalse(actual.values.flags.writeable)
                    self.assertEqual(bound.repetition_counts_by_participant[owner], (3 + number,) + (1,) * 223)
                    self.assertIn((inputs.READ_OWNER, owner), env.admitted)
                self.assertEqual([(path.parent.name, path.name) for path in env.paths],
                    [(owner, name) for owner in inputs.PARTICIPANTS
                     for name in ("tau_test_responses.npy", "tau_test_finite_packed.npy")])
                self.assertTrue(all(array._mmap.closed for array in env.maps))
                with self.assertRaises(TypeError):
                    bound.responses_by_participant["sub-01"] = bound.responses_by_participant["sub-03"]
                bound.truth_reader._check()
                self.assertEqual(env.admitted[-1], (inputs.READ_TRUTH, None))
                bound.check()
                self.assertEqual(env.admitted[-1], (inputs.BIND, None))
                env.budget.allocation["job_id"] = "moved-job"
                with self.assertRaisesRegex(ValueError, "another_allocation"):
                    bound.check()
            finally:
                bound.close()
            self.assertTrue(env.readers[0]._closed)

    def test_refusals_before_values_include_last_owner_metadata_and_trial_roster(self):
        with environment() as env:
            with mock.patch.object(inputs, "_metadata") as metadata, \
                    mock.patch.object(inputs, "_load_array") as load:
                denied = dict(env.kwargs, require_admitted_context=mock.Mock(side_effect=PermissionError("closed")))
                with self.assertRaises(PermissionError):
                    inputs.bind_full_finding_a_inputs(**denied)
                metadata.assert_not_called()
                load.assert_not_called()
            self.assertEqual(env.readers, [])
        for defect in ("last_width", "last_nonfinite_metadata", "missing_owner", "last_trial_role",
                       "last_trial_owner", "duplicate_last_trial", "common_owner_swap"):
            with self.subTest(defect=defect), environment() as env, ExitStack() as stack:
                if defect == "last_width":
                    env.neural["subjects"][-1]["roles"]["tau_test"]["voxels"] -= 1
                elif defect == "last_nonfinite_metadata":
                    env.neural["subjects"][-1]["roles"]["tau_test"]["nonfinite_elements"] = 1
                elif defect == "missing_owner":
                    env.neural["subjects"].pop()
                elif defect == "common_owner_swap":
                    wrong = replace(env.sources["sub-03"], root=env.sources["sub-01"].root)
                    stack.enter_context(mock.patch.object(inputs, "_common_metadata", return_value=(wrong, env.truth)))
                else:
                    rows = [dict(row) for row in env.rows["sub-07"]]
                    if defect == "last_trial_role":
                        rows[-1]["role"] = "tau_train"
                    elif defect == "last_trial_owner":
                        rows[-1]["participant"] = "sub-03"
                    else:
                        rows[-1]["beta_volume_index"] = rows[0]["beta_volume_index"]
                    table(env.sources["sub-07"].root / "tau_test_trials.tsv", rows)
                env.neural_path.write_text(json.dumps(env.neural))
                with mock.patch.object(inputs, "_load_array") as load, self.assertRaises(ValueError):
                    inputs.bind_full_finding_a_inputs(**env.kwargs)
                load.assert_not_called()
                self.assertEqual(env.readers, [])
                self.assertEqual(env.maps, [])

    def test_nonfinite_revocation_cancellation_and_map_reader_cleanup(self):
        for defect in ("nonfinite_value", "false_finite_bit", "mean_overflow", "revoked_owner", "cancel_between_maps"):
            with self.subTest(defect=defect), environment() as env:
                owner = "sub-07"
                root = env.sources[owner].root
                expected_error = ValueError
                if defect == "nonfinite_value":
                    env.values[owner][0, 0] = np.nan
                    np.save(root / "tau_test_responses.npy", env.values[owner])
                elif defect == "false_finite_bit":
                    finite = np.ones(env.values[owner].shape, dtype=np.uint8)
                    finite[0, 0] = 0
                    np.save(root / "tau_test_finite_packed.npy", np.packbits(finite, axis=1, bitorder="little"))
                elif defect == "mean_overflow":
                    env.values[owner][list(env.sources[owner].groups[0]), 0] = np.finfo(np.float32).max
                    np.save(root / "tau_test_responses.npy", env.values[owner])
                elif defect == "revoked_owner":
                    admitted = env.kwargs["require_admitted_context"]
                    def revoked(actual, **kwargs):
                        admitted(actual, **kwargs)
                        if kwargs["operation"] == inputs.READ_OWNER and kwargs["source"] == "sub-05":
                            raise PermissionError("owner revoked")
                    env.kwargs["require_admitted_context"] = revoked
                    expected_error = PermissionError
                else:
                    def cancel():
                        if (env.paths and env.paths[-1].parent.name == "sub-03"
                                and env.paths[-1].name == "tau_test_responses.npy"):
                            raise TimeoutError("allocation ended")
                    env.budget.check = cancel
                    expected_error = TimeoutError
                with self.assertRaises(expected_error):
                    inputs.bind_full_finding_a_inputs(**env.kwargs)
                self.assertEqual(len(env.readers), 1)
                self.assertTrue(env.readers[0]._closed)
                self.assertTrue(all(array._mmap.closed for array in env.maps))
                self.assertFalse(any(path.name.startswith("tau_train") for path in env.paths))


if __name__ == "__main__":
    unittest.main()
