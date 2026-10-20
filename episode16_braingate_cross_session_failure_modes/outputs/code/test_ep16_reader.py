"""Focused checks of the EP16 role seal and neural-window reader on synthetic MAT files."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

import ep16_roles
from ep16_roles import RoleSealError, SealedEvaluation, authorize_protected_read
from mat5_neural import read_session_windows


def fixture(participant="T11", compressed=True, bins=200, electrodes=3):
    rng = np.random.default_rng(0)
    sbp = rng.normal(size=(bins, electrodes))
    sbp[45, 1] = np.nan  # invalidates channel 1 for the trial starting at 40
    tx = rng.integers(0, 4, size=(bins, electrodes)).astype(float)
    cursor = rng.normal(size=(bins, 2))
    target = rng.normal(size=(bins, 2))
    fields = {"participant_id": participant, "post_implant_day": 104, "bin_size": .01,
              "electrodes": {"electrode_id": np.arange(electrodes), "array": np.array(["lateral"] * electrodes, dtype=object),
                             "x": np.arange(electrodes), "y": np.zeros(electrodes), "impedance": np.ones(electrodes)},
              "cursor_position": cursor, "target_position": target,
              "blocks": {"block_id": [1, 2], "sample_start_index": [0, 100], "sample_end_index": [100, 200]},
              "trials": {"trial_start_index": [5, 40, 80, 120, 185], "trial_end_index": [30, 70, 99, 150, 199],
                         "block_id": [1, 1, 1, 2, 2]},
              "neural": {"sbp": sbp, "tx_3": tx, "tx_4_5": tx * 2}}
    out = io.BytesIO()
    savemat(out, fields, do_compression=compressed)
    return out, fields


class ReaderAndSealTests(unittest.TestCase):
    def test_window_means_cursor_rows_and_invalid_channels(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                source, f = fixture(compressed=compressed)
                values, out, index, access = read_session_windows(source, "T11")
                starts = np.array([5, 40, 80, 120, 185])
                np.testing.assert_array_equal(index["window_inside_block"], [True, True, False, True, False])
                kept = index["whole"] & index["window_inside_block"]  # cursor rows only for whole, inside trials
                np.testing.assert_allclose(out["cursor_start"][kept], f["cursor_position"][starts[kept]])
                self.assertTrue(np.isnan(out["cursor_start"][~kept]).all())
                for t in (0, 3):
                    np.testing.assert_allclose(out["sbp"][t], f["neural"]["sbp"][starts[t]:starts[t] + 30].mean(axis=0))
                    np.testing.assert_allclose(out["tx_4_5"][t], f["neural"]["tx_4_5"][starts[t]:starts[t] + 30].mean(axis=0))
                    np.testing.assert_allclose(out["sbp_odd_even"][0, t], f["neural"]["sbp"][starts[t]:starts[t] + 30:2].mean(axis=0))
                self.assertTrue(np.isnan(out["sbp"][1, 1]))
                self.assertFalse(np.isnan(out["sbp"][1, [0, 2]]).any())
                self.assertTrue(np.isnan(out["sbp"][[2, 4]]).all())  # window crosses the block end
                self.assertIn("neural.sbp", access["protected_decoded_paths"])
                self.assertNotIn("neural.tx_3", access["protected_decoded_paths"])
                self.assertIn("neural.tx_3", access["skipped_paths"])
                self.assertNotIn("electrodes.impedance", access["structural"]["decoded_paths"])

    def test_seal_refuses_before_decoding(self):
        for participant in ("T9", "T2", "T6", "T7", "T8", "T99"):
            with self.subTest(participant=participant):
                source, _ = fixture(participant=participant)
                with self.assertRaises(RoleSealError):
                    read_session_windows(source, participant)
        source, _ = fixture(participant="T9")
        with self.assertRaises(RoleSealError):  # archive says development, file says T9
            read_session_windows(source, "T11")
        with self.assertRaises(RoleSealError):
            SealedEvaluation(object(), "T9", ".", "x")

    def test_sealed_token_requires_matching_opened_record(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ep16_roles, "EVALUATION_ROOT", Path(tmp)):
            attempt = Path(tmp) / "attempt01"  # tokens are valid only for a direct attemptNN child
            attempt.mkdir()
            token = SealedEvaluation(ep16_roles._ISSUER, "T9", attempt, "nonce-1")
            with self.assertRaises(RoleSealError):
                authorize_protected_read("T9", token)
            (attempt / ep16_roles.OPENED_RECORD).write_text(json.dumps({"nonce": "nonce-1", "participant": "T9"}))
            self.assertEqual(authorize_protected_read("T9", token), "sealed_evaluation")
            with self.assertRaises(RoleSealError):  # a T9 token cannot open development data
                authorize_protected_read("T11", token)
            source, _ = fixture(participant="T9")
            values, out, _, _ = read_session_windows(source, "T9", token=token)
            self.assertEqual(out["sbp"].shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
