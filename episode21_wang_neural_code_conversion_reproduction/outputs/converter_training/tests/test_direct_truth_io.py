"""Synthetic direct-reader checks; coordinator-owned compute execution only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finding_a_direct_truth as direct
import finding_a_empirical_inputs as inputs
import test_finding_a_empirical_inputs as original_tests


class DirectTruthIOTests(unittest.TestCase):
    def test_original_coordinates_flags_bank_bound_and_headers(self):
        with mock.patch.object(inputs, "TruthReader", direct.DirectTruthReader):
            original_tests.FindingAEmpiricalInputsTests.test_truth_C_coordinates_packed_offsets_bounded_banks_and_headers(self)

    def test_both_handles_use_zero_sieve_readonly(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(inputs, "SHAPES", original_tests.SHAPES):
            root = Path(directory).resolve()
            original_tests.truth_fixture(root)
            with direct.DirectTruthReader(root, image_ids=original_tests.IDS,
                    feature_identity=original_tests.IDENTITY, check=lambda: None) as reader:
                reader(layer=inputs.LAYERS[0], image_ids=original_tests.IDS,
                    full_layer_C_index=np.array([35]), feature_identity=original_tests.IDENTITY)
                handles = reader._features, reader._flags
                for handle in handles:
                    self.assertEqual(handle.mode, "r")
                    properties = handle.id.get_access_plist()
                    try:
                        self.assertEqual(properties.get_sieve_buf_size(), 0)
                    finally:
                        properties.close()
            self.assertTrue(all(not handle.id.valid for handle in handles))

    def test_second_hdf_open_failure_closes_first(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(inputs, "SHAPES", original_tests.SHAPES):
            root = Path(directory).resolve()
            original_tests.truth_fixture(root)
            opener, opened = direct.open_truth_readonly, []

            def second_failure(path):
                if Path(path).name == "finite_packed.h5":
                    raise OSError("fixture second HDF")
                handle = opener(path)
                opened.append(handle)
                return handle

            reader = direct.DirectTruthReader(root, image_ids=original_tests.IDS,
                feature_identity=original_tests.IDENTITY, check=lambda: None)
            with mock.patch.object(direct, "open_truth_readonly", side_effect=second_failure), \
                    self.assertRaisesRegex(OSError, "fixture second HDF"):
                reader(layer=inputs.LAYERS[0], image_ids=original_tests.IDS,
                    full_layer_C_index=np.array([35]), feature_identity=original_tests.IDENTITY)
            self.assertEqual(len(opened), 1)
            self.assertFalse(opened[0].id.valid)
            self.assertTrue(reader._closed)
            self.assertIsNone(reader._features)
            self.assertIsNone(reader._flags)

    def test_admission_revocation_prevents_reads_and_closes_handles(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(inputs, "SHAPES", original_tests.SHAPES):
            root = Path(directory).resolve()
            original_tests.truth_fixture(root)
            revoked = True

            def check():
                if revoked:
                    raise PermissionError("fixture revoked")

            request = dict(layer=inputs.LAYERS[0], image_ids=original_tests.IDS,
                full_layer_C_index=np.array([35]), feature_identity=original_tests.IDENTITY)
            reader = direct.DirectTruthReader(root, image_ids=original_tests.IDS,
                feature_identity=original_tests.IDENTITY, check=check)
            with mock.patch.object(direct, "open_truth_readonly") as unopened, \
                    self.assertRaisesRegex(PermissionError, "fixture revoked"):
                reader(**request)
            unopened.assert_not_called()
            self.assertTrue(reader._closed)

            revoked = False
            reader = direct.DirectTruthReader(root, image_ids=original_tests.IDS,
                feature_identity=original_tests.IDENTITY, check=check)
            reader(**request)
            handles = reader._features, reader._flags
            revoked = True
            with mock.patch.object(h5py.Dataset, "__getitem__",
                    side_effect=AssertionError("payload read after revocation")) as unread, \
                    mock.patch.object(direct, "open_truth_readonly") as unopened, \
                    self.assertRaisesRegex(PermissionError, "fixture revoked"):
                reader(**request)
            unread.assert_not_called()
            unopened.assert_not_called()
            self.assertTrue(reader._closed)
            self.assertTrue(all(not handle.id.valid for handle in handles))
            self.assertIsNone(reader._bank)
            self.assertIsNone(reader._packed)


if __name__ == "__main__":
    unittest.main()
