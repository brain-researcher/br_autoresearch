"""Bounded source-free I/O fixtures; run only in the coordinator's allocation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_interface as interface


LAYER = "conv5_1"
SHAPE = (3, 5, 7)
CHANNEL = 1
CAP = interface.IMAGES * SHAPE[1] * 2 * 4


def fixture_shape(layer, channel):
    if (layer, channel) == (LAYER, CHANNEL):
        return SHAPE, SHAPE[1] * SHAPE[2]
    if (layer, channel) == ("fc8", None):
        return (11,), 11
    raise ValueError("fixture layer/channel")


def block(start, count, *, layer=LAYER, channel=CHANNEL):
    spatial = np.arange(start, start + count, dtype=np.int64)
    full_f = spatial if channel is None else channel + SHAPE[0] * spatial
    full_c = (spatial if channel is None else channel * SHAPE[1] * SHAPE[2]
              + (spatial % SHAPE[1]) * SHAPE[2] + spatial // SHAPE[1])
    return interface.DecoderBlock("sub-01", layer, channel, start, full_f, full_c,
        np.zeros((count, 1), dtype=np.int64), np.zeros((count, 1), dtype=np.int64),
        np.ones((count, 1), dtype=np.float32), np.zeros(count, dtype=np.float32),
        spatial.astype(np.float32) * np.float32(.125) - np.float32(1),
        (spatial % 5).astype(np.float32) + np.float32(.75))


def fixture():
    raw = np.arange(interface.IMAGES * np.prod(SHAPE), dtype=np.float32)
    raw = raw.reshape((interface.IMAGES,) + SHAPE) / np.float32(64) - np.float32(2000)
    bits = np.ones((interface.IMAGES, int(np.prod(SHAPE))), dtype=np.uint8)
    # Neighboring channels share boundary bytes; their false flags are irrelevant.
    bits[:, 0] = bits[:, 34] = bits[:, 70] = 0
    packed = np.packbits(bits, axis=1, bitorder="little")
    blocks = tuple(block(start, size) for start, size in ((0, 7), (7, 6), (13, 10), (23, 12)))
    return raw, packed, blocks


def write_fixture(directory, raw, packed):
    root = Path(directory) / "outputs/steward_provisioning/regular_vgg19_sealed/sub-03"
    root.mkdir(parents=True)
    with h5py.File(root / "features.h5", "w") as handle:
        handle.create_dataset(LAYER, data=raw, chunks=None, compression=None)
    with h5py.File(root / "finite_packed.h5", "w") as handle:
        handle.create_dataset(LAYER, data=packed, chunks=None, compression=None)
    return root


class RecordingDataset:
    def __init__(self, data, *, stripe=False):
        self.data, self.shape = data, data.shape
        self.reads, self.sizes = [], []
        self.stripe = stripe
        self.previous = None

    def __getitem__(self, key):
        if self.stripe and self.previous is not None and self.previous() is not None:
            raise AssertionError("previous stripe must be released before another read")
        result = self.data[key]
        self.reads.append(key)
        self.sizes.append(result.nbytes)
        if self.stripe:
            self.previous = weakref.ref(result)
        return result


class FeatureStripeTests(unittest.TestCase):
    def test_contiguous_full_rows_exact_oracle_boundaries_and_two_passes(self):
        self.assertEqual(interface.FEATURE_STRIPE_BYTES, 67_108_864)
        raw, packed, blocks = fixture()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(interface, "active_shape", side_effect=fixture_shape), \
                mock.patch.object(interface, "FEATURE_STRIPE_BYTES", CAP):
            root = write_fixture(directory, raw, packed)
            with h5py.File(root / "features.h5", "r") as values, \
                    h5py.File(root / "finite_packed.h5", "r") as masks:
                data = RecordingDataset(values[LAYER], stripe=True)
                flags = RecordingDataset(masks[LAYER])
                old_data = RecordingDataset(values[LAYER])
                for pass_number in range(2):
                    reader = interface._FeatureStripeReader(data, flags,
                        layer=LAYER, channel=CHANNEL, check=lambda: None)
                    try:
                        for item in blocks:
                            actual = reader.read(item)
                            oracle = interface._read_feature_units(old_data, masks[LAYER], item, lambda: None)
                            units = range(item.unit_start, item.unit_start + len(item.bias))
                            direct = np.stack([raw[:, CHANNEL, unit % 5, unit // 5]
                                               for unit in units], axis=1)
                            direct -= item.target_mean
                            direct /= item.target_std
                            self.assertEqual(actual.tobytes(), oracle.tobytes())
                            self.assertEqual(actual.tobytes(), direct.tobytes())
                            self.assertEqual(actual.shape[0], 4712)
                            self.assertFalse(actual.flags.writeable)
                        self.assertEqual(len(data.reads), 4 * (pass_number + 1))
                        self.assertEqual(len(flags.reads), pass_number + 1)
                    finally:
                        reader.close()
                    self.assertIsNone(reader._stripe)
                    self.assertIsNone(reader._packed)
                    self.assertIsNone(data.previous())
                self.assertEqual(data.reads[:4], [
                    (slice(None), CHANNEL, slice(None), slice(start, min(start + 2, 7)))
                    for start in (0, 2, 4, 6)])
                self.assertEqual(flags.reads, [(slice(None), slice(4, 9))] * 2)
                self.assertTrue(all(size <= CAP for size in data.sizes))
                self.assertEqual(data.sizes[3], CAP // 2)
                self.assertGreater(len(old_data.reads), len(data.reads))

            fc_values = np.arange(4712 * 11, dtype=np.float32).reshape(4712, 11)
            fc_flags = np.full((4712, 2), 255, dtype=np.uint8)
            fc = block(3, 7, layer="fc8", channel=None)
            reader = interface._FeatureStripeReader(fc_values, fc_flags,
                layer="fc8", channel=None, check=lambda: None)
            with mock.patch.object(interface, "_read_feature_units", wraps=interface._read_feature_units) as thin:
                actual = reader.read(fc)
                thin.assert_called_once()
            expected = (fc_values[:, 3:10] - fc.target_mean) / fc.target_std
            np.testing.assert_array_equal(actual, expected)
            self.assertIsNone(reader._stripe)
            self.assertIsNone(reader._packed)
            reader.close()

    def test_pass_cleanup_on_exhaustion_explicit_close_and_callback_cancellation(self):
        raw, packed, blocks = fixture()
        original_reader = interface._FeatureStripeReader
        with tempfile.TemporaryDirectory() as directory:
            root = write_fixture(directory, raw, packed)
            pair = SimpleNamespace(source=SimpleNamespace(
                axis=SimpleNamespace(participant="sub-03"), feature_root=root))
            for mode in ("exhaust", "close", "cancel"):
                with self.subTest(mode=mode):
                    created, closed = [], []

                    def make_reader(*args, **kwargs):
                        reader = original_reader(*args, **kwargs)
                        created.append(reader)
                        return reader

                    def decoder_blocks():
                        try:
                            yield from blocks
                        finally:
                            closed.append(True)

                    def check():
                        if mode == "cancel" and created and created[-1]._stripe is not None:
                            raise TimeoutError("fixture cancellation after stripe read")

                    with mock.patch.object(interface.assembly, "BASE", Path(directory)), \
                            mock.patch.object(interface, "active_shape", side_effect=fixture_shape), \
                            mock.patch.object(interface, "FEATURE_STRIPE_BYTES", CAP), \
                            mock.patch.object(interface, "_FeatureStripeReader", side_effect=make_reader):
                        content = interface._iter_content_blocks(pair, layer=LAYER, channel=CHANNEL,
                            decoder_blocks=decoder_blocks(), check=check)
                        if mode == "cancel":
                            with self.assertRaisesRegex(TimeoutError, "after stripe read"):
                                next(content)
                        elif mode == "close":
                            next(content)
                            self.assertIsNotNone(created[0]._stripe)
                            content.close()
                        else:
                            self.assertEqual(len(list(content)), len(blocks))
                    self.assertEqual(closed, [True])
                    self.assertEqual(len(created), 1)
                    reader = created[0]
                    self.assertIsNone(reader._stripe)
                    self.assertIsNone(reader._packed)
                    self.assertIsNone(reader.data)
                    self.assertIsNone(reader.finite)
                    with self.assertRaisesRegex(ValueError, "closed"):
                        reader.read(blocks[0])

    def test_refusals_match_thin_reader_without_checking_unrequested_prefetch(self):
        original, original_flags, _ = fixture()
        early, later = block(0, 5), block(5, 5)
        bad_unit = 7
        c_index = CHANNEL * 35 + (bad_unit % 5) * 7 + bad_unit // 5
        for defect, message in (("flag", "source-feature flag"),
                                ("value", "nonfinite required source feature"),
                                ("scale", "fixed positive target-decoder"),
                                ("overflow", "nonfinite target-standardized")):
            with self.subTest(defect=defect), \
                    mock.patch.object(interface, "active_shape", side_effect=fixture_shape), \
                    mock.patch.object(interface, "FEATURE_STRIPE_BYTES", CAP), \
                    np.errstate(over="ignore", invalid="ignore"):
                raw, flags = original.copy(), original_flags.copy()
                item = later
                if defect == "flag":
                    flags[4711, c_index // 8] &= np.uint8(255 ^ (1 << (c_index % 8)))
                elif defect == "value":
                    raw[4711, CHANNEL, bad_unit % 5, bad_unit // 5] = np.nan
                elif defect == "scale":
                    item = replace(later, target_std=np.zeros(5, dtype=np.float32))
                else:
                    item = replace(later, target_std=np.full(5, np.finfo(np.float32).tiny, dtype=np.float32))
                reader = interface._FeatureStripeReader(raw, flags,
                    layer=LAYER, channel=CHANNEL, check=lambda: None)
                try:
                    # The first stripe also contains unit7; refusal belongs to
                    # that unit's block, not to unrelated read-ahead bytes.
                    np.testing.assert_array_equal(reader.read(early),
                        interface._read_feature_units(raw, flags, early, lambda: None))
                    with self.assertRaisesRegex(ValueError, message) as optimized:
                        reader.read(item)
                    with self.assertRaisesRegex(ValueError, message) as thin:
                        interface._read_feature_units(raw, flags, item, lambda: None)
                    self.assertEqual(str(optimized.exception), str(thin.exception))
                finally:
                    reader.close()
                self.assertIsNone(reader._stripe)
                self.assertIsNone(reader._packed)


if __name__ == "__main__":
    unittest.main()
