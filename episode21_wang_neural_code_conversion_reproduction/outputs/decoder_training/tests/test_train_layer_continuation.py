"""Focused new multi-channel checks, only inside a substantive Slurm allocation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_layer_continuation as driver


class LayerContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        driver.prefix.load_libraries()
        cls.np, cls.h5py = driver.prefix.np, driver.prefix.h5py

    def budget(self):
        return driver.Budget({"job_end_epoch": time.time() + 86400}, driver.MAX_SECONDS)

    def test_new_authority_before_allocation_imports_or_payload(self):
        with mock.patch.object(driver.prefix, "read_json") as read:
            with self.assertRaisesRegex(driver.Stop, "fixed_layer_authorization_path"):
                driver.authorize(driver.prior.AUTH)
            read.assert_not_called()
        for change in ({"approved": False}, {"approved": 1}, {"channel_interval": [0, 64]},
                       {"resources": driver.prior.RESOURCES}, {"reuse_completed_units": 1216},
                       {"scratch_cap_interpretation": "cumulative_bytes_written"}):
            with mock.patch.object(driver.prefix, "read_json", return_value=dict(driver.AUTH_EXPECTED, **change)):
                with self.assertRaisesRegex(driver.Stop, "explicit_new_layer_authorization"):
                    driver.authorize(driver.AUTH)
        with mock.patch.object(Path, "cwd", return_value=driver.BASE), \
                mock.patch.object(driver.prefix, "read_json", side_effect=FileNotFoundError), \
                mock.patch.object(driver.prior, "verify_allocation") as allocation, \
                mock.patch.object(driver.prefix, "load_libraries") as libraries:
            with self.assertRaises(FileNotFoundError):
                driver.run(SimpleNamespace(authorization=driver.AUTH, max_seconds=driver.MAX_SECONDS))
            allocation.assert_not_called()
            libraries.assert_not_called()

    def test_full_target_exact_exclusion_and_no_false_assembly(self):
        coverage = driver.coverage
        retained = list(coverage.retained_spans())
        remaining = list(coverage.remaining_spans())
        self.assertEqual(sum(s.count for s in retained), 51328)
        self.assertEqual(sum(s.count for s in remaining), 74255112)
        self.assertEqual(coverage.verify_complete_coverage(retained + remaining), 74306440)
        self.assertEqual(sum(s.count for s in driver.SPANS), 3161088)
        self.assertTrue(all(s.channel > 0 and s.participant == "sub-01" and s.layer == "conv1_1"
                            for s in driver.SPANS))
        old_layer = [s for s in retained if s.layer == "conv1_1"]
        self.assertEqual(coverage.verify_complete_coverage(old_layer + list(driver.SPANS),
                            participants=("sub-01",), layers=("conv1_1",)), 3211264)
        with self.assertRaisesRegex(driver.Stop, "missing_required_channels"):
            coverage.verify_complete_coverage(retained + list(driver.SPANS))
        for defective in (retained + remaining + retained[:1], retained[1:] + remaining):
            with self.assertRaisesRegex(driver.Stop, "coverage_gap_or_overlap"):
                coverage.verify_complete_coverage(defective)
        work = list(coverage.pack_work(remaining))
        self.assertTrue(75 <= len(work) <= 80)
        self.assertTrue(all(sum(s.count for s in item) <= 1003520
                            and len({s.participant for s in item}) == 1 for item in work))
        self.assertEqual([s for item in work for s in item], remaining)

    def test_nonzero_channel_plane_and_unaligned_packed_flags(self):
        np = self.np
        # A small 14x14 channel plane exercises channel starts inside a byte,
        # unlike conv1's byte-aligned channels. No empirical file is opened.
        shape = (512, 14, 14)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(driver.prefix, "IMAGES", 3):
            root = Path(temporary)
            data = np.arange(3 * 512 * 14 * 14, dtype=np.float32).reshape((3,) + shape)
            valid = np.ones((3, 512 * 14 * 14), dtype=np.uint8)
            # Flag outside the selected channel must not leak into its check.
            valid[0, 195] = 0
            with self.h5py.File(root / "features.h5", "w") as h5:
                h5.create_dataset("conv5_1", data=data)
            with self.h5py.File(root / "finite_packed.h5", "w") as h5:
                h5.create_dataset("conv5_1", data=np.packbits(valid, axis=1, bitorder="little"))
            for channel in (1, 511):
                span = driver.coverage.Span("sub-01", "conv5_1", channel, 0, 196)
                bank = driver.stage_channel(root, root, span, self.budget())
                try:
                    np.testing.assert_array_equal(bank, data[:, channel].transpose(0, 2, 1).reshape(3, 196))
                    for spatial in (0, 1, 13, 14, 195):
                        coordinate, full_f, full_c = driver.coverage.coordinates(shape, channel, spatial)
                        self.assertEqual(full_f, channel + 512 * spatial)
                        self.assertEqual(float(bank[0, spatial]), full_c)
                        self.assertEqual(coordinate, (channel, spatial % 14, spatial // 14))
                finally:
                    bank._mmap.close()
                    (root / "channel.float32").unlink()
            valid[1, 196 + 14] = 0
            with self.h5py.File(root / "finite_packed.h5", "r+") as h5:
                h5["conv5_1"][:] = np.packbits(valid, axis=1, bitorder="little")
            with self.assertRaisesRegex(driver.Stop, "image=1:spatial_F=1"):
                driver.stage_channel(root, root, driver.coverage.Span("sub-01", "conv5_1", 1, 0, 196), self.budget())

    def test_grouped_data_before_markers_no_replay_and_uncertain_marker_flush(self):
        np = self.np
        span = driver.coverage.Span("sub-01", "conv1_1", 63, 0, 4)
        row = (np.arange(500), np.ones(500, dtype=np.float32), 2., 3., 4., .1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "channels.h5"
            with self.h5py.File(path, "x") as container:
                store = driver.ChannelStore(container, path, span, self.budget())
                np.testing.assert_array_equal(store.file["full_layer_Fortran_index"][:], [63, 127, 191, 255])
                np.testing.assert_array_equal(store.file["full_layer_C_index"][:], 63 * 50176 + np.arange(4) * 224)
                real_sync = driver.prefix.fsync_h5
                seen = []
                def sync(handle):
                    self.assertIs(handle, container)
                    seen.append(store.file["state"][:].tolist())
                    real_sync(handle)
                with mock.patch.object(driver.prefix, "fsync_h5", side_effect=sync):
                    store.commit(0, [row, row])
                self.assertEqual(seen, [[0, 0, 0, 0], [2, 2, 0, 0]])
                with self.assertRaisesRegex(driver.Stop, "disjoint_ascending"):
                    store.commit(0, [row])
                with mock.patch.object(driver.prefix, "fsync_h5", side_effect=OSError("data failure")):
                    with self.assertRaisesRegex(OSError, "data failure"):
                        store.commit(2, [row])
                self.assertEqual(store.committed_end, 2)
                self.assertIsNone(store.uncertain_publication)
                calls = []
                def marker_failure(handle):
                    calls.append(True)
                    if len(calls) == 2:
                        raise OSError("marker failure")
                    real_sync(handle)
                with mock.patch.object(driver.prefix, "fsync_h5", side_effect=marker_failure):
                    with self.assertRaisesRegex(OSError, "marker failure"):
                        store.commit(2, [row])
                self.assertEqual(store.committed_end, 2)
                self.assertEqual(store.uncertain_publication["channel"], 63)
                self.assertEqual(store.uncertain_publication["spatial_Fortran_interval"], [2, 3])
                self.assertEqual(store.uncertain_publication["phase"], "completion_marker_fsync")
                self.assertEqual(int(store.file["state"][2]), 2)

    def test_fixed_math_float32_layout_short_tail_and_partial_stop(self):
        np = self.np
        span = driver.coverage.Span("sub-01", "conv1_1", 1, 0, 65)
        bank = np.arange(5 * 65, dtype=np.float32).reshape(5, 65)
        native = np.arange(501)
        calls = []
        def prepare(x, augmented, gram, targets, corrmat):
            self.assertTrue(targets.flags.c_contiguous)
            self.assertEqual(targets.dtype, np.dtype("float32"))
            begin = 0 if not calls else 64
            np.testing.assert_array_equal(targets, bank[:, begin:begin + targets.shape[1]])
            count = targets.shape[1]
            calls.append(count)
            return {"branch": "source_greater_than_500_with_native_ID_ties",
                    "right": np.zeros((count, 502)), "correlations": np.zeros((count, 501)),
                    "means": np.zeros((1, count), dtype=np.float32),
                    "scales": np.ones((1, count), dtype=np.float32)}
        def solve(gram, right, correlations, ids):
            self.assertIs(ids, native)
            return np.arange(500), np.ones(500, dtype=np.float32), np.float32(2)
        timing = {k: 0. for k in ("normalization_correlation_RHS_seconds", "select_solve_seconds", "publication_seconds")}
        store, budget = mock.Mock(), self.budget()
        with mock.patch.object(driver.prefix, "prepare_layer_math", side_effect=prepare), \
                mock.patch.object(driver.prefix, "solve_selected", side_effect=solve):
            driver.fit_blocks(bank, span, None, None, None, native, None, store, budget, timing)
        self.assertEqual(calls, [64, 1])
        self.assertEqual([(call.args[0], len(call.args[1])) for call in store.commit.call_args_list], [(0, 64), (64, 1)])
        calls.clear()
        store.reset_mock()
        with mock.patch.object(driver.prefix, "prepare_layer_math", side_effect=prepare), \
                mock.patch.object(driver.prefix, "solve_selected", side_effect=solve), \
                mock.patch.object(budget, "check", side_effect=[None, None, None, driver.Stop("deadline")]):
            with self.assertRaisesRegex(driver.Stop, "deadline"):
                driver.fit_blocks(bank, span, None, None, None, native, None, store, budget, timing)
        self.assertEqual(len(store.commit.call_args.args[1]), 2)

    def test_closeout_attempts_every_action_after_fragment_or_cleanup_failure(self):
        container, bank, budget = mock.Mock(), mock.Mock(), mock.Mock()
        container.close.side_effect = OSError("close failure")
        budget.note.side_effect = OSError("stat failure")
        bank._mmap.close.side_effect = OSError("bank failure")
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            (scratch / "channel.float32").write_bytes(b"test")
            with mock.patch.object(driver.prefix, "fsync_h5", side_effect=OSError("fsync failure")):
                errors = driver.finalize_files(container, bank, scratch, Path(temporary) / "fragment.h5", budget)
            self.assertEqual([row["stage"] for row in errors],
                             ["fragment_fsync", "fragment_close", "fragment_accounting", "channel_bank_close"])
            self.assertEqual([row["affects_fragment"] for row in errors], [True, True, True, False])
            self.assertFalse(scratch.exists())
            container.close.assert_called_once()
            budget.note.assert_called_once()
            bank._mmap.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
