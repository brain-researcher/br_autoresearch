"""Three bounded cache checks; prepared only, for coordinated Slurm execution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import content_channel_cache as cache_module
from content_loss import ActiveLayer, LossBlock, content_loss_and_adjoint
from test_content_loss import fixture


class ArrayDataset:
    def __init__(self, values):
        self.values, self.shape, self.dtype = values, values.shape, values.dtype
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return np.array(self.values[key], copy=True, order="C")


def arrays(block):
    return (block.eligible_columns, block.weights, block.bias, block.standardized_features)


def numpy_fixture():
    layers, blocks = fixture()
    return layers, {name: [LossBlock(block.unit_start, *(value.numpy().copy() for value in arrays(block)))
                          for block in records] for name, records in blocks.items()}


def tiny_cache(provider, layers, *, check=lambda: None, check_pass=lambda: None,
               maximum_bytes=4096):
    return cache_module._TwoPassCache(provider, check_pass=check_pass, check=check,
        layer_order=tuple(active.layer for active in layers), expected_rows=3,
        validate_active=lambda active: None, maximum_bytes=maximum_bytes)


class ContentChannelCacheTests(unittest.TestCase):
    def test_selected_channel_F_order_nonbyte_flags_and_exact_normalization(self):
        # Channel 1 starts at C bit 15, deliberately not a byte boundary.
        shape, rows = (3, 3, 5), 4
        raw = np.arange(rows * np.prod(shape), dtype=np.float32).reshape((rows,) + shape)
        raw[:, 0] = np.nan
        raw[:, 2] = np.nan
        valid = np.zeros((rows, np.prod(shape)), dtype=np.uint8)
        valid[:, 15:30] = 1
        data, flags = ArrayDataset(raw), ArrayDataset(np.packbits(valid, axis=1, bitorder="little"))

        def block(first, count):
            units = np.arange(first, first + count, dtype=np.int64)
            return SimpleNamespace(layer="conv_fixture", channel=1, unit_start=first,
                bias=np.zeros(count, dtype=np.float32),
                full_layer_C_index=np.array([15 + (int(unit) % 3) * 5 + int(unit) // 3
                                             for unit in units], dtype=np.int64),
                full_layer_Fortran_index=1 + 3 * units,
                target_mean=np.arange(count, dtype=np.float32) - .25,
                target_std=np.arange(count, dtype=np.float32) + .5)

        with mock.patch.object(cache_module.interface, "IMAGES", rows), \
             mock.patch.object(cache_module.interface, "active_shape", return_value=(shape, 15)):
            reader = cache_module.ContiguousChannelReader(data, flags, layer="conv_fixture",
                                                         channel=1, check=lambda: None)
            self.assertEqual(data.reads + flags.reads, [])
            for first, count in ((0, 4), (4, 5), (9, 6)):
                current = block(first, count)
                expected = np.stack([raw[row, 1].reshape(-1, order="F") for row in range(rows)])
                expected = expected[:, first:first + count].copy()
                expected -= current.target_mean
                expected /= current.target_std
                observed = reader.read(current)
                np.testing.assert_array_equal(observed, expected)
                self.assertFalse(observed.flags.writeable)
            self.assertEqual(data.reads, [(slice(None), 1, slice(None), slice(None))])
            self.assertEqual(flags.reads, [(slice(None), slice(1, 4))])
            self.assertEqual(reader.retained_bytes, 252)
            reader.close()
            self.assertEqual(reader.retained_bytes, 0)

            for failure in ("coordinates", "flag", "value", "cap"):
                changed = ArrayDataset(raw.copy())
                marked = ArrayDataset(flags.values.copy())
                current = block(0, 4)
                if failure == "coordinates":
                    current.full_layer_C_index[0] += 1
                elif failure == "flag":
                    marked.values[0, 15 // 8] &= np.uint8(255 ^ (1 << (15 % 8)))
                elif failure == "value":
                    changed.values[0, 1, 0, 0] = np.nan
                with self.subTest(failure=failure), \
                     mock.patch.object(cache_module, "MAX_CHANNEL_BYTES", 251 if failure == "cap" else 1024**3), \
                     self.assertRaises(ValueError):
                    trial = cache_module.ContiguousChannelReader(changed, marked,
                        layer="conv_fixture", channel=1, check=lambda: None)
                    trial.read(current)
                if failure in ("coordinates", "cap"):
                    self.assertEqual(changed.reads + marked.reads, [])
                else:
                    self.assertEqual(trial.retained_bytes, 0)

        # The same reader handles FC units without inventing a channel axis.
        fc = ArrayDataset(np.arange(rows * 7, dtype=np.float32).reshape(rows, 7))
        ff = ArrayDataset(np.full((rows, 1), 127, dtype=np.uint8))
        current = SimpleNamespace(layer="fc_fixture", channel=None, unit_start=2,
            bias=np.zeros(3, dtype=np.float32), full_layer_C_index=np.arange(2, 5),
            full_layer_Fortran_index=np.arange(2, 5), target_mean=np.ones(3, dtype=np.float32),
            target_std=np.full(3, 2, dtype=np.float32))
        with mock.patch.object(cache_module.interface, "IMAGES", rows), \
             mock.patch.object(cache_module.interface, "active_shape", return_value=((7,), 7)):
            reader = cache_module.ContiguousChannelReader(fc, ff, layer="fc_fixture",
                                                         channel=None, check=lambda: None)
            np.testing.assert_array_equal(reader.read(current), (fc.values[:, 2:5] - 1) / 2)
            self.assertEqual(fc.reads, [(slice(None), slice(None))])
            reader.close()

    def test_exact_two_pass_arrays_and_loss_adjoint_with_one_upstream_pass(self):
        layers, blocks = numpy_fixture()
        calls, checked = [], []

        def provider(active):
            calls.append(active)
            return iter(blocks[active.layer])

        cached = tiny_cache(provider, layers, check_pass=lambda: checked.append("admitted"))
        first = [list(cached(active)) for active in layers]
        original_bytes = [[tuple(value.tobytes() for value in arrays(block)) for block in records]
                          for records in first]
        expected_bytes = sum(value.nbytes for records in first for block in records for value in arrays(block))
        self.assertEqual(cached.retained_bytes, expected_bytes)
        for records in blocks.values():
            for block in records:
                for value in arrays(block):
                    value[:] = 0  # Incoming arrays cannot mutate the retained owning copies.
        second = [list(cached(active)) for active in layers]
        for index, (before, after) in enumerate(zip(first, second)):
            for position, (left, right) in enumerate(zip(before, after)):
                self.assertIs(left, right)
                self.assertEqual(tuple(value.tobytes() for value in arrays(right)), original_bytes[index][position])
                for value in arrays(right):
                    self.assertTrue(value.flags.owndata)
                    self.assertFalse(value.flags.writeable)
        self.assertEqual(calls, list(layers))
        self.assertEqual(len(checked), 8)
        self.assertEqual((cached.retained_bytes, cached.peak_retained_bytes, cached.completed_iterations),
                         (0, expected_bytes, 1))

        layers, blocks = numpy_fixture()
        values = torch.tensor([[.2, -.3, .4, .1, .5], [.8, .2, -.1, .9, .3],
                               [-.5, .7, .6, -.2, 1.]], dtype=torch.float32)
        original = content_loss_and_adjoint(values, layers, lambda active: iter(blocks[active.layer]),
            expected_rows=3, max_block_units=2, row_chunk_size=2, backend="dense")
        calls.clear()
        cached = tiny_cache(provider, layers)
        observed = content_loss_and_adjoint(values, layers, cached, expected_rows=3,
            max_block_units=2, row_chunk_size=2, backend="dense")
        for name in ("loss", "norms", "layer_weights", "layer_squared_errors", "response_adjoint"):
            self.assertTrue(torch.equal(getattr(observed, name), getattr(original, name)), name)
        self.assertEqual(calls, list(layers))
        self.assertEqual((cached.retained_bytes, cached.completed_iterations), (0, 1))

        # Exercise the PUBLIC constructor against the actual training schedule,
        # not an independently injected layer order. Only fixture row/unit sizes
        # are small; all19 identities and the next iteration's channels are real.
        from schedule import make_channel_schedule
        actual_schedule = make_channel_schedule()
        scheduled_calls = []
        one_unit = blocks["fc_fixture"][0]

        def scheduled_provider(active):
            scheduled_calls.append((active.layer, active.channel))
            return iter((one_unit,))

        with mock.patch.object(cache_module.interface, "IMAGES", 3), \
                mock.patch.object(cache_module.interface, "active_layer_units", return_value=1):
            public = cache_module.TwoPassLossCache(scheduled_provider,
                                                   check_pass=lambda: None, check=lambda: None)
            for iteration in (0, 1):
                declared = actual_schedule.blocks(iteration)
                active = tuple(ActiveLayer(layer, channel, 1) for layer, channel in declared)
                norm_blocks = [list(public(item)) for item in active]
                self.assertEqual(scheduled_calls, [entry for step in range(iteration + 1)
                                                   for entry in actual_schedule.blocks(step)])
                gradient_blocks = [list(public(item)) for item in active]
                self.assertEqual(len(scheduled_calls), (iteration + 1) * len(declared))
                for before, after in zip(norm_blocks, gradient_blocks):
                    self.assertIs(before[0], after[0])
                self.assertEqual((public.retained_bytes, public.completed_iterations), (0, iteration + 1))
            self.assertNotEqual(actual_schedule.blocks(0), actual_schedule.blocks(1))
            public.close()

    def test_byte_cap_exhaustion_order_revocation_and_partial_close_poison(self):
        layers, blocks = numpy_fixture()
        active = layers[1]
        original = blocks[active.layer][0]
        closed, calls = [], []

        def provider(request):
            calls.append(request)
            try:
                yield original
            finally:
                closed.append(request)

        # Construction/iterator creation does no work; last yield is NOT exhaustion.
        cached = tiny_cache(provider, (active,))
        pending = cached(active)
        self.assertEqual(calls, [])
        next(pending)
        self.assertGreater(cached.retained_bytes, 0)
        pending.close()
        self.assertEqual((cached.retained_bytes, cached.completed_iterations), (0, 0))
        self.assertEqual(closed, [active])
        with self.assertRaisesRegex(ValueError, "closed_or_failed"):
            list(cached(active))

        class NoCopy(np.ndarray):
            def copy(self, *args, **kwargs):
                raise AssertionError("byte cap must precede allocating any copy")

        oversized = LossBlock(original.unit_start, *(value.view(NoCopy) for value in arrays(original)))
        size = sum(value.nbytes for value in arrays(original))
        cached = tiny_cache(lambda _: iter((oversized,)), (active,), maximum_bytes=size - 1)
        with self.assertRaisesRegex(ValueError, "byte_cap_before_copy"):
            list(cached(active))
        self.assertEqual(cached.retained_bytes, 0)

        for failure in ("order", "changed_replay", "revoked_after_exhaustion", "cancel", "external_close"):
            closed.clear()
            checks = []
            cancelled = [False]

            def check_pass():
                checks.append(1)
                if failure == "revoked_after_exhaustion" and len(checks) == 2:
                    raise PermissionError("original binding revoked")

            def check():
                if cancelled[0]:
                    raise TimeoutError("bounded cancellation")

            cached = tiny_cache(provider, (active,), check_pass=check_pass, check=check)
            with self.subTest(failure=failure):
                if failure == "order":
                    with self.assertRaisesRegex(ValueError, "pass_order"):
                        list(cached(layers[0]))
                elif failure == "changed_replay":
                    list(cached(active))
                    with self.assertRaisesRegex(ValueError, "same_iteration"):
                        list(cached(replace(active, channel=1)))
                elif failure == "revoked_after_exhaustion":
                    with self.assertRaises(PermissionError):
                        list(cached(active))
                else:
                    pending = cached(active)
                    next(pending)
                    if failure == "cancel":
                        cancelled[0] = True
                        with self.assertRaises(TimeoutError):
                            next(pending)
                    else:
                        cached.close()
                        self.assertEqual(closed, [active])
                        pending.close()
                self.assertEqual((cached.retained_bytes, cached.completed_iterations), (0, 0))
                if failure != "order":
                    self.assertEqual(closed, [active])
                with self.assertRaisesRegex(ValueError, "closed_or_failed"):
                    list(cached(active))


if __name__ == "__main__":
    unittest.main()
