"""Focused scratch-cap/lifecycle fixtures; run only in root's Slurm allocation.

No production inputs or live scratch paths are opened. Temporary arrays are
small fixtures for cap enforcement and one-bank cleanup, not scientific fits.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_decoder_work_item as worker

prefix, layer = worker.prefix, worker.layer


class WorkerScratchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix.load_libraries()
        cls.np, cls.h5py = prefix.np, prefix.h5py

    def resources(self):
        return {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
                "durable_bytes": 8_000_000_000, "scratch_bytes": 2_000_000_000}

    def spans(self):
        return (worker.contract.coverage.Span("sub-01", "conv1_1", 2, 256, 512),
                worker.contract.coverage.Span("sub-01", "fc8", 0, 960, 1000))

    def test_effective_cap_binds_full_channel_shape_and_keeps_manifest_and_legacy_resources(self):
        original = self.resources()
        saved = deepcopy(original)
        spans = self.spans()
        # A partial interval still stages the entire spatial channel bank.
        expected = 4712 * 224 * 224 * 4 + 64 * 1024**2
        admitted = {"effective_scratch_bytes": expected,
                    "parallelism_amendment_id": "parallelism_amendment"}
        effective = worker.runtime_resources(original, spans, admitted)
        self.assertEqual(effective, {**saved, "scratch_bytes": expected})
        self.assertEqual(original, saved)
        self.assertIsNot(effective, original)
        self.assertEqual(worker.runtime_resources(original, spans, None), saved)
        fc = (worker.contract.coverage.Span("sub-07", "fc8", 0, 960, 1000),)
        self.assertEqual(worker.largest_channel_bank_bytes(fc), 4712 * 1000 * 4)

    def test_unbound_or_nonexact_amended_cap_is_rejected_without_resource_widening(self):
        original = self.resources()
        spans = self.spans()
        expected = 4712 * 224 * 224 * 4 + 64 * 1024**2
        correct = {"effective_scratch_bytes": expected,
                   "parallelism_amendment_id": "parallelism_amendment"}
        policies = [False, {}, {**correct, "extra": 1},
                    {**correct, "parallelism_amendment_id": "unreviewed"}]
        policies += [{**correct, "effective_scratch_bytes": value}
                     for value in (True, str(expected), expected - 1, expected + 1, 2_000_000_001)]
        for policy in policies:
            with self.subTest(policy=policy), self.assertRaises(prefix.Stop):
                worker.runtime_resources(original, spans, policy)
        with self.assertRaises(prefix.Stop):
            worker.runtime_resources({**original, "scratch_bytes": expected - 1}, spans, correct)
        self.assertEqual(original, self.resources())

    def test_bank_cap_fails_before_mmap_capacity_probe_or_feature_open(self):
        span = worker.contract.coverage.Span("sub-07", "fc8", 0, 960, 1000)
        amount = 4712 * 1000 * 4
        for cap in (amount - 1, 0, True, 2_000_000_001):
            with self.subTest(cap=cap), \
                 mock.patch.object(prefix.np, "memmap") as allocate, \
                 mock.patch.object(prefix.h5py, "File") as source, \
                 mock.patch.object(layer.os, "statvfs") as capacity:
                with self.assertRaisesRegex(prefix.Stop, "scratch_cap"):
                    layer.stage_channel(Path("fixture"), Path("fixture"), span,
                                        SimpleNamespace(), scratch_bytes=cap)
                allocate.assert_not_called()
                source.assert_not_called()
                capacity.assert_not_called()

    def test_one_bank_is_reused_only_after_cleanup_and_partial_staging_is_closed(self):
        np, h5py = self.np, self.h5py
        span = worker.contract.coverage.Span("sub-07", "fc8", 0, 0, 1000)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(prefix, "IMAGES", 2):
            root = Path(temporary)
            scratch = root / "worker"
            scratch.mkdir()
            values = np.arange(2000, dtype=np.float32).reshape(2, 1000)
            with h5py.File(root / "features.h5", "w") as source:
                source.create_dataset("fc8", data=values)
            with h5py.File(root / "finite_packed.h5", "w") as source:
                source.create_dataset("fc8", data=np.packbits(np.ones((2, 1000), dtype=np.uint8),
                                                              axis=1, bitorder="little"))
            budget = SimpleNamespace(check=lambda: None, logical_read_bytes=0)
            # Old callers retain their positional API and original 2-GB cap.
            bank = layer.stage_channel(root, scratch, span, budget)
            np.testing.assert_array_equal(bank, values)
            self.assertEqual((scratch / "channel.float32").stat().st_size, values.nbytes)
            with self.assertRaisesRegex(prefix.Stop, "fresh_channel_bank"):
                layer.stage_channel(root, scratch, span, budget, scratch_bytes=values.nbytes)
            bank._mmap.close()
            (scratch / "channel.float32").unlink()
            bank = layer.stage_channel(root, scratch, span, budget, scratch_bytes=values.nbytes)
            np.testing.assert_array_equal(bank, values)
            errors = layer.finalize_files(None, bank, scratch, root / "unused.h5", budget)
            self.assertEqual(errors, [])
            self.assertTrue(bank._mmap.closed)
            self.assertFalse(scratch.exists())

            # A failure before the bank is returned still closes its mapping;
            # ordinary finalization removes only that worker's owned scratch.
            scratch.mkdir()
            allocated = []
            actual_memmap = np.memmap
            def allocate(*args, **kwargs):
                result = actual_memmap(*args, **kwargs)
                allocated.append(result)
                return result
            with mock.patch.object(np, "memmap", side_effect=allocate), \
                 mock.patch.object(h5py, "File", side_effect=OSError(5, "fixture source error")):
                with self.assertRaises(OSError):
                    layer.stage_channel(root, scratch, span, budget, scratch_bytes=values.nbytes)
            self.assertEqual(len(allocated), 1)
            self.assertTrue(allocated[0]._mmap.closed)
            self.assertTrue((scratch / "channel.float32").exists())
            errors = layer.finalize_files(None, None, scratch, root / "unused.h5", budget)
            self.assertEqual(errors, [])
            self.assertFalse(scratch.exists())
            self.assertTrue((root / "features.h5").is_file())
            self.assertTrue((root / "finite_packed.h5").is_file())


if __name__ == "__main__":
    unittest.main()
