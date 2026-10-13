"""Focused source-free data fixtures for the new prefix adapter; run only in Slurm.

These checks have not been executed during preparation. They do not repeat the
completed public VGG/converter conformance or access empirical payloads.
"""
from pathlib import Path
import json
import os
import tempfile
import time
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_retained_prefix as driver


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        driver.load_libraries()
        cls.np = driver.np

    def test_authorization_scope_precedes_any_read(self):
        with mock.patch.object(driver, "read_json") as reader:
            with self.assertRaisesRegex(driver.Stop, "fixed_authorization_path"):
                driver.authorize("/tmp/outside_authorization.json")
            reader.assert_not_called()

    def test_early_ordering_requires_explicit_boolean_and_matching_whitelist(self):
        current = dict(driver.AUTH_EXPECTED, complete_feature_export_required=True)
        with mock.patch.object(driver, "read_json", return_value=current):
            self.assertIs(driver.authorize(driver.AUTH), current)
        current[driver.EARLY_AUTHORITY] = False
        with mock.patch.object(driver, "read_json", return_value=current):
            self.assertIs(driver.authorize(driver.AUTH), current)
        for value in ("true", 1, None):
            record = dict(current, **{driver.EARLY_AUTHORITY: value})
            with self.subTest(value=value), mock.patch.object(driver, "read_json", return_value=record):
                with self.assertRaisesRegex(driver.Stop, "boolean_ordering_authority"):
                    driver.authorize(driver.AUTH)
        amended = dict(current, **{driver.EARLY_AUTHORITY: True})
        with mock.patch.object(driver, "read_json", return_value=amended):
            with self.assertRaisesRegex(driver.Stop, "explicit_ordering_amendment"):
                driver.authorize(driver.AUTH)
        amended["complete_feature_export_required"] = False
        with mock.patch.object(driver, "read_json", return_value=amended):
            with self.assertRaisesRegex(driver.Stop, "explicit_decoder_prefix_authorization"):
                driver.authorize(driver.AUTH)
        amended["feature_inputs"] = [str((driver.PRESERVED_FEATURES / name).relative_to(driver.BASE))
                                     for name in driver.FEATURE_FILES]
        with mock.patch.object(driver, "read_json", return_value=amended):
            self.assertIs(driver.authorize(driver.AUTH), amended)
        amended[driver.EARLY_AUTHORITY] = False
        with mock.patch.object(driver, "read_json", return_value=amended):
            with self.assertRaisesRegex(driver.Stop, "approved_complete_export_order"):
                driver.authorize(driver.AUTH)

    def test_default_requires_complete_global_report_without_preserved_fallback(self):
        lock = {"feature_recipe": "fixture"}
        groups = {"sub-01": [1], "tau_train": [2]}
        report = {"publication": "complete", "feature_recipe": "fixture",
                  "layer_shapes": dict(zip(driver.LAYERS, map(list, driver.SHAPES))),
                  "feature_dtype": "float32", "OOD_access": False,
                  "partitions": [{"partition": name, "images": 1, "nonfinite_elements": 0}
                                 for name in groups]}
        with mock.patch.object(driver, "read_json", side_effect=FileNotFoundError) as reader:
            with self.assertRaises(FileNotFoundError):
                driver.admit_feature_export(lock, groups)
            reader.assert_called_once_with(driver.PROV / "regular_vgg19_structural_report.json")
        with mock.patch.object(driver, "read_json", return_value=dict(report, publication="incomplete")):
            with self.assertRaisesRegex(driver.Stop, "complete_regular_feature_export"):
                driver.admit_feature_export(lock, groups)
        with mock.patch.object(driver, "read_json", return_value=dict(report, partitions=report["partitions"][:1])):
            with self.assertRaisesRegex(driver.Stop, "complete_frozen_feature_partitions"):
                driver.admit_feature_export(lock, groups)
        with mock.patch.object(driver, "read_json", return_value=report) as reader:
            source, admission = driver.admit_feature_export(lock, groups)
            self.assertEqual(source, driver.FEATURES)
            self.assertTrue(admission["global_feature_export_completion_claimed"])
            reader.assert_called_once_with(driver.PROV / "regular_vgg19_structural_report.json")

    def test_preserved_source_stays_bound_when_separate_publication_tree_is_renamed(self):
        # Tiny stand-in files reproduce the continuation's filesystem operation,
        # never its feature contents. No real partition is inspected or linked.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            prov = base / "outputs/steward_provisioning"
            original = prov / "regular_vgg19_sealed.building.46424955/sub-01"
            final = prov / "regular_vgg19_sealed/sub-01"
            lock = base / "outputs/protocol_lock.yaml"
            original.mkdir(parents=True)
            for name in driver.FEATURE_FILES:
                (original / name).write_text("filesystem fixture: " + name)
            history = prov / "regular_vgg19_attempt003-46424955.out"
            records = [{"partition": subject, "status": "real_regular_features_provisioned_sealed"}
                       for subject in ("sub-01", "sub-03")]
            records.append({"publication": "incomplete", "error_type": "RuntimeError",
                            "error_code": "deadline_reached"})
            history.write_text("fixture progress\n" + "\n".join(map(json.dumps, records)) + "\n")
            auth_path = base / "outputs/regular_vgg19_runtime_retry_authorization_2026-10-03.json"
            claim_path = prov / "regular_vgg19_export_job_claim.json"
            metadata = {auth_path: {"approved": True, "purpose": "regular_vgg19_derivative_export"},
                        claim_path: {"job_id": "46424955", "purpose": "regular_vgg19_derivative_export",
                                     "protocol_lock": str(lock)}}
            with mock.patch.multiple(driver, BASE=base, PROV=prov, LOCK=lock,
                                     PRESERVED_FEATURES=original, FEATURES=final), \
                    mock.patch.object(driver, "read_json", side_effect=lambda p: metadata[p]) as reader:
                pinned, admission = driver.admit_feature_export({}, {}, early_start=True)
                self.assertEqual(pinned, original)
                self.assertFalse(admission["global_feature_export_completion_claimed"])
                identities = {name: (pinned / name).stat().st_ino for name in driver.FEATURE_FILES}
                publishing = prov / "regular_vgg19_sealed.continuation-20261005.building"
                (publishing / "sub-01").mkdir(parents=True)
                for name in driver.FEATURE_FILES:
                    os.link(original / name, publishing / "sub-01" / name)
                publishing.rename(final.parent)
                self.assertEqual(driver.feature_source(early_start=True), original)
                self.assertEqual(driver.feature_source(), final)
                for name, inode in identities.items():
                    self.assertEqual((pinned / name).stat().st_ino, inode)
                    self.assertEqual((final / name).stat().st_ino, inode)
                    self.assertEqual((pinned / name).read_text(), "filesystem fixture: " + name)
                # Canonical publication cannot rescue missing original provenance
                # or a missing originally admitted file.
                metadata[claim_path]["job_id"] = "another_job"
                with self.assertRaisesRegex(driver.Stop, "original_feature_claim"):
                    driver.admit_feature_export({}, {}, early_start=True)
                metadata[claim_path]["job_id"] = "46424955"
                history.write_text("\n".join(map(json.dumps, records[1:])) + "\n")
                with self.assertRaisesRegex(driver.Stop, "original_sub01_sealed_completion"):
                    driver.admit_feature_export({}, {}, early_start=True)
                history.write_text("\n".join(map(json.dumps, records)) + "\n")
                (original / "features.h5").unlink()
                with self.assertRaisesRegex(driver.Stop, "preserved_sub01_three_file_source"):
                    driver.admit_feature_export({}, {}, early_start=True)
                self.assertTrue((final / "features.h5").is_file())
                self.assertTrue(all(call.args[0] in metadata for call in reader.call_args_list))

    def test_preserved_headers_and_index_require_every_locked_layer_and_row(self):
        h5py = driver.h5py
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            index = source / "image_index.tsv"
            header = "output_row_index\tregular_HDF5_row\timage_id\tparticipant\trole\n"
            rows = ["0\t3\ta\tsub-01\tsubject_unique\n", "1\t7\tb\tsub-01\tsubject_unique\n"]
            index.write_text(header + "".join(rows))
            selected = [(3, {"image_id": "a"}), (7, {"image_id": "b"})]
            with h5py.File(source / "features.h5", "x") as f, h5py.File(source / "finite_packed.h5", "x") as g:
                for name, shape in (("conv", (2, 2, 2)), ("fc", (9,))):
                    f.create_dataset(name, shape=(2,) + shape, dtype="<f4")
                    g.create_dataset(name, shape=(2, (driver.math.prod(shape) + 7) // 8), dtype="u1")
            lock = {"runtime": {"h5py": h5py.__version__}}
            with mock.patch.multiple(driver, LAYERS=("conv", "fc"), SHAPES=((2, 2, 2), (9,)), IMAGES=2):
                self.assertEqual(driver.feature_index(index, selected), ["a", "b"])
                driver.preserved_feature_headers(source, lock, lambda: None)
                index.write_text(header + rows[0] + rows[1].replace("subject_unique", "tau_train"))
                with self.assertRaisesRegex(driver.Stop, "feature_index_role_or_identity"):
                    driver.feature_index(index, selected)
                index.write_text(header + rows[0])
                with self.assertRaisesRegex(driver.Stop, "feature_index_role_or_identity"):
                    driver.feature_index(index, selected)
                index.write_text(header + "".join(rows) + rows[0])
                with self.assertRaisesRegex(driver.Stop, "complete_unique_feature_rows"):
                    driver.feature_index(index, selected)
                with h5py.File(source / "features.h5", "r+") as f:
                    del f["fc"]
                    f.create_dataset("fc", shape=(2, 9), dtype="<f8")
                with self.assertRaisesRegex(driver.Stop, "preserved_feature_layer_header"):
                    driver.preserved_feature_headers(source, lock, lambda: None)
                with h5py.File(source / "features.h5", "r+") as f:
                    del f["fc"]
                with self.assertRaisesRegex(driver.Stop, "complete_locked_layer_vocabulary"):
                    driver.preserved_feature_headers(source, lock, lambda: None)

    def test_all_fixed_units_keep_full_tensor_identity(self):
        rows = [r for layer, shape in zip(driver.LAYERS, driver.SHAPES)
                for r in driver.unit_specs(layer, shape)]
        self.assertEqual(len(rows), 1216)
        conv = driver.unit_specs("conv5_4", (512, 14, 14))
        self.assertEqual(conv[63]["tensor_coordinate"], [0, 7, 4])
        self.assertEqual(conv[63]["full_layer_Fortran_index"], 512 * 63)
        self.assertEqual(conv[63]["full_layer_C_index"], 102)
        self.assertEqual(rows[-1]["tensor_coordinate"], [63])
        self.assertTrue(all(row["status"] == "pending" for row in rows))

    def test_repetition_join_preserves_all_rows_and_equal_image_means(self):
        np = self.np
        rows = [{"output_row_index": str(i), "participant": "sub-01", "session": "s",
                 "beta_volume_index": str(i), "image_id": image, "role": "subject_unique"}
                for i, image in enumerate(("a", "b", "a", "b"))]
        groups = driver.group_trials(rows, ["b", "a"], 4)
        self.assertEqual(groups, [[1, 3], [0, 2]])
        values = np.asarray([[1, 2], [10, 20], [3, 6], [14, 28]], dtype=np.float32)
        flags = np.packbits(np.ones(values.shape, dtype=np.uint8), axis=1, bitorder="little")
        means, counts = driver.aggregate_images(values, flags, groups, lambda: None)
        np.testing.assert_array_equal(means, [[12, 24], [2, 4]])
        np.testing.assert_array_equal(counts, [2, 2])
        self.assertEqual(means.dtype, np.float32)
        contaminated = [dict(row) for row in rows]
        contaminated[0]["role"] = "tau_train"
        with self.assertRaisesRegex(driver.Stop, "role_or_row_identity"):
            driver.group_trials(contaminated, ["b", "a"], 4)

    def test_zero_input_variance_and_undefined_target_have_distinct_rules(self):
        np = self.np
        values = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 501))
        values[:, -1] = 7
        x, ids, _, _, eligible = driver.normalize_neural(values, np.arange(501))
        self.assertEqual(x.shape, (4, 500))
        self.assertFalse(eligible[-1])
        np.testing.assert_array_equal(ids, np.arange(500))
        augmented, gram = driver.source_gram(x)
        targets = np.column_stack((np.arange(4, dtype=np.float32), np.ones(4, dtype=np.float32)))
        result = driver.prepare_layer_math(x, augmented, gram, targets, lambda *a: self.fail("must stop before correlation"))
        self.assertEqual(result["bad_units"], [1])
        values[:, -2] = 8
        with self.assertRaisesRegex(driver.Stop, "fewer_than_500"):
            driver.normalize_neural(values, np.arange(501))

    def test_exact500_branch_and_greater500_ties_use_source_ridge(self):
        np = self.np
        base = np.arange(8, dtype=np.float32)
        base = (base - np.mean(base)) / np.std(base, ddof=1)
        x = np.tile(base[:, None], (1, 502)).astype(np.float32)
        augmented, gram = driver.source_gram(x)
        right = np.matmul(base[None, :], augmented)[0]
        indices, weights, bias = driver.solve_selected(gram, right, np.ones(502, dtype=np.float32), np.arange(502))
        np.testing.assert_array_equal(indices, np.arange(500))
        expected = float(np.sum(base * base)) / (100 + 500 * float(np.sum(base * base)))
        np.testing.assert_allclose(weights, expected, rtol=2e-4, atol=1e-7)
        self.assertAlmostEqual(float(bias), 0.0, places=6)
        x = x[:, :500].copy()
        augmented, gram = driver.source_gram(x)
        result = driver.prepare_layer_math(x, augmented, gram, base[:, None],
                                          lambda *a: self.fail("exactly500 must not rank"))
        self.assertEqual(result["branch"], "source_exactly_500_native_order_no_selection")
        np.testing.assert_allclose(result["solved"][:500, 0], expected, rtol=2e-4, atol=1e-7)
        self.assertAlmostEqual(float(result["solved"][-1, 0]), 0.0, places=6)

    def test_HDF5_plane_reader_keeps_Fortran_units_and_rejects_nonfinite_flag(self):
        np, h5py = self.np, driver.h5py
        shape = (2, 4, 16)
        values = np.arange(3 * 128, dtype=np.float32).reshape((3,) + shape)
        flags = np.packbits(np.ones((3, 128), dtype=np.uint8), axis=1, bitorder="little")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            budget = driver.Budget({"job_end_epoch": time.time() + 900}, 100)
            with h5py.File(path / "values.h5", "x") as f, h5py.File(path / "flags.h5", "x") as g:
                f.create_dataset("layer", data=values)
                g.create_dataset("layer", data=flags)
                with mock.patch.object(driver, "IMAGES", 3):
                    target, layout = driver.read_layer(f, g, "layer", shape, driver.unit_specs("layer", shape), budget)
                    np.testing.assert_array_equal(target, np.stack([row[0].reshape(-1, order="F") for row in values]))
                    self.assertIsNone(layout["chunks"])
                    g["layer"][0, 0] = 0
                    with self.assertRaises(driver.UnitStop) as raised:
                        driver.read_layer(f, g, "layer", shape, driver.unit_specs("layer", shape), budget)
                    self.assertIn(0, raised.exception.indices)

    def test_completed_unit_cannot_be_rewritten_and_other_units_stay_pending(self):
        np, h5py = self.np, driver.h5py
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fc8.h5"
            budget = driver.Budget({"job_end_epoch": time.time() + 900}, 100)
            specs = driver.unit_specs("fc8", (1000,))
            store = driver.LayerStore(path, specs, budget)
            try:
                store.start(0)
                store.complete(0, np.arange(500), np.ones(500, dtype=np.float32), 0.0, 1.0, 2.0, 0.01)
                with self.assertRaisesRegex(driver.Stop, "already_started"):
                    store.start(0)
                with self.assertRaisesRegex(driver.Stop, "cannot_change_completed"):
                    store.fail([0], "replacement")
                self.assertEqual(specs[0]["status"], "complete")
                self.assertTrue(all(row["status"] == "pending" for row in specs[1:]))
            finally:
                store.close()
            with h5py.File(path, "r") as saved:
                self.assertEqual(int(saved["state"][0]), 2)
                np.testing.assert_array_equal(saved["state"][1:], np.zeros(63, dtype=np.uint8))
                np.testing.assert_array_equal(saved["selected_native_voxel_ids"][0], np.arange(500))
            with self.assertRaises(FileExistsError):
                driver.LayerStore(path, driver.unit_specs("fc8", (1000,)), budget)


if __name__ == "__main__":
    unittest.main()
