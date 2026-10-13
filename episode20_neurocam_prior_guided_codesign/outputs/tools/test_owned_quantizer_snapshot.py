"""Three finite ownership/parity checks; window check requires reuse patch.

No production schedule, saved world, model fitting or electrical solve is used.
"""

from dataclasses import replace
from fractions import Fraction
import hashlib
import struct
import unittest
from unittest.mock import patch

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from owned_quantizer_snapshot import snapshot_quantizer
from streaming_acquisition_readout import ReadoutWindow, StreamingAcquisitionReadout, StreamingReadoutLimits
from streaming_array_acquisition import ArrayWindow, _canonical, _window_digest


def transfer():
    return QuantizerTransfer(-1., 1., (-.5, -0.0, .5), (-.75, -.25, .25, .75))


def scalar_digest(value):
    digest = hashlib.sha256()
    digest.update(struct.pack("<ddQQ", value.input_min_volts, value.input_max_volts,
                              len(value.thresholds_volts), len(value.code_volts)))
    for sequence in (value.thresholds_volts, value.code_volts):
        for item in sequence:
            digest.update(struct.pack("<d", item))
    return digest.hexdigest()


class OwnedQuantizerSnapshotTests(unittest.TestCase):
    def test_full_admission_then_constant_size_distinct_snapshots_preserve_values(self):
        caller = transfer()
        thresholds, codes = list(caller.thresholds_volts), list(caller.code_volts)
        # The ordinary input path must still own/sanitize mutable sequences,
        # even when callers reassigned them after constructing a frozen object.
        object.__setattr__(caller, "thresholds_volts", thresholds)
        object.__setattr__(caller, "code_volts", codes)
        expected = transfer()
        owned = snapshot_quantizer(caller)
        self.assertIs(type(owned), QuantizerTransfer)
        self.assertEqual(owned, expected)
        self.assertIs(type(owned.thresholds_volts), tuple)
        self.assertTrue(all(type(item) is float for item in (*owned.thresholds_volts, *owned.code_volts)))
        self.assertIs(type(owned.input_min_volts), float)
        self.assertIs(type(owned.input_max_volts), float)
        thresholds[0], codes[0] = -.9, -.95
        self.assertEqual(owned, expected)
        with patch.object(QuantizerTransfer, "__post_init__",
                          side_effect=AssertionError("admitted immutable values were revalidated")):
            for _ in range(4):
                copied = snapshot_quantizer(owned)
                self.assertIsNot(copied, owned)
                self.assertIs(copied.thresholds_volts, owned.thresholds_volts)
                self.assertIs(copied.code_volts, owned.code_volts)
                self.assertEqual(copied, expected)
                self.assertEqual(scalar_digest(copied), scalar_digest(expected))
                for voltage in (-2., -1., -.5, -0.0, 0., .5, 1., 2.):
                    self.assertEqual(copied.convert(voltage), expected.convert(voltage))
                owned = copied

    def test_reassigned_fields_rejected_and_provenance_free_inputs_fully_validated(self):
        for name in ("input_min_volts", "input_max_volts", "thresholds_volts", "code_volts"):
            owned = snapshot_quantizer(transfer())
            peer = snapshot_quantizer(owned)
            original = getattr(owned, name)
            replacement = tuple(list(original)) if type(original) is tuple else float(str(original))
            self.assertIsNot(replacement, original)
            object.__setattr__(owned, name, replacement)
            with self.assertRaisesRegex(ValueError, "reassigned"):
                snapshot_quantizer(owned)
            self.assertEqual(snapshot_quantizer(peer), transfer())
        zero = snapshot_quantizer(QuantizerTransfer(-0.0, 1., (.5,), (.25, .75)))
        object.__setattr__(zero, "input_min_volts", +0.0)
        with self.assertRaisesRegex(ValueError, "reassigned"):
            snapshot_quantizer(zero)
        # Explicit replacement is a new input, not inherited provenance.
        replaced = replace(snapshot_quantizer(transfer()), code_volts=(-.7, -.25, .25, .75))
        self.assertFalse(hasattr(replaced, "_validated_quantizer_payload"))
        self.assertEqual(snapshot_quantizer(replaced), replaced)
        for bad in (True, float("nan")):
            ordinary = replace(replaced)
            object.__setattr__(ordinary, "input_min_volts", bad)
            with self.assertRaises(ValueError):
                snapshot_quantizer(ordinary)
        ordinary = replace(replaced)
        object.__setattr__(ordinary, "code_volts", [0., -.25, .25, .75])
        with self.assertRaises(ValueError):
            snapshot_quantizer(ordinary)
        fractional = QuantizerTransfer(0., Fraction(1, 10), (.05,), (.025, .075))
        copied = snapshot_quantizer(fractional)
        self.assertIs(copied.input_max_volts, fractional.input_max_volts)
        self.assertEqual(copied.convert(.1), fractional.convert(.1))
        self.assertTrue(copied.convert(.1)[2])
        self.assertFalse(hasattr(copied, "_validated_quantizer_payload"))
        class TaggedTransfer(QuantizerTransfer):
            def convert(self, volts):
                index, value, clipped = super().convert(volts)
                return index, value + 1., clipped
        tagged = TaggedTransfer(-1., 1., (-.5, 0., .5), (-.75, -.25, .25, .75))
        copied = snapshot_quantizer(tagged)
        self.assertIs(type(copied), TaggedTransfer)
        self.assertIsNot(copied, tagged)
        self.assertEqual(copied.convert(.1), tagged.convert(.1))
        self.assertFalse(hasattr(copied, "_validated_quantizer_payload"))

    def test_window_template_and_reader_reuse_keep_isolation_and_fresh_digest(self):
        quantizer = transfer()
        raw = RawConversion("r:0", "adc", "input", "signal", .1, .2, .2, .1, quantizer, True)
        sample = ReturnedSample("s:0", ("r:0",), (1.,), 1., .1, .3, quantizer)
        observation = Observation("o:0", "site", "pixel", ("s:0",), (1.,), 1., .1, .4, 8, None)
        first = ReadoutWindow(0, 0., 1., (raw,), (sample,), (observation,),
                              (("r:0", 0),), (("s:0", 0),))
        selected = first.conversions[0].quantizer
        self.assertIs(selected, first.returned_samples[0].output_quantizer)
        self.assertIsNot(selected, quantizer)
        self.assertEqual(_canonical(selected, {}), {"quantizer_value_sha256": scalar_digest(quantizer)})
        digest = lambda window: _window_digest(ArrayWindow(0, 0., 1., (), window), b"")
        original_digest = digest(first)
        reader = StreamingAcquisitionReadout(input_ids=("input",), acquisition_start=0.,
            acquisition_end=1., final_publication_horizon=1.,
            limits=StreamingReadoutLimits(10, 10, 10, 100, 10, 4, 4, 4, 4096, 8),
            on_raw=lambda _: None, on_observation=lambda _: None, validate_only=True)
        with patch.object(QuantizerTransfer, "__post_init__",
                          side_effect=AssertionError("template/window/readout performed full validation")):
            # These are the actual schedule/template and runtime snapshot seams.
            template = replace(first)
            generated = replace(template, conversions=(replace(template.conversions[0]),))
            self.assertEqual(digest(generated), original_digest)
            reader.accept_window(generated)
            runtime = reader._raw["r:0"].quantizer
            self.assertIsNot(runtime, generated.conversions[0].quantizer)
            self.assertIs(runtime.thresholds_volts, selected.thresholds_volts)
        changed = replace(selected, code_volts=(-.7, -.25, .25, .75))
        replay = replace(first, conversions=(replace(first.conversions[0], quantizer=changed),))
        self.assertNotEqual(digest(replay), original_digest)
        # Reassignment of the caller's earlier snapshot cannot mutate runtime.
        object.__setattr__(selected, "code_volts", changed.code_volts)
        self.assertEqual(runtime, quantizer)
        self.assertNotEqual(digest(first), original_digest)
        with self.assertRaisesRegex(ValueError, "reassigned"):
            replace(first)


if __name__ == "__main__":
    unittest.main()
