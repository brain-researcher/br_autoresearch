"""Three complete-roster mismatch checks; run only in coordinated compute.

Small fabricated native packs test pairing and access invariants, not a
scientific known-answer case or empirical score. The pure records_for fixture
is reused from test_native_trials; no reader, model or scorer is invoked.
"""
from dataclasses import replace
import hashlib
import unittest

import numpy as np

import fit_mismatch as core
from native_trials import EARLIER, _snapshot, adapt_records
from test_native_trials import records_for


def check():
    pass


def fixture():
    result = []
    for region in ("M1", "PMd"):
        rows = []
        for index, date in enumerate(EARLIER["Mihili"] + ("20140304",)):
            role = "calibration" if index == 3 else "training"
            current = records_for(date, role, region,
                neurons=index + 1 + (region == "PMd"),
                first_trial=999998 if index == 3 else 999990,
                spike_offset=100 * index, electrode=7 + index)
            for record in current:
                # Distinct native-bin/band values catch partial-trajectory moves.
                record["lfp"] += (np.arange(len(record["bin_offsets"]))[:, None] / 32
                                  + np.arange(9)[None, :] / 512).astype(np.float32)
            current.reverse()
            rows.extend(current)
            if index == 3:
                calibration = current
        scope = dict(view_role="fit", animal="Mihili", target_day="20140304",
                     region=region, check=check)
        result.extend(adapt_records(rows, stream="history_assisted", **scope))
        result.extend(adapt_records(calibration, stream="today_only", **scope))
    return tuple(result)


def expected_donors(day, seed):
    """Scalar ID-to-ID oracle for the frozen key/endianness/shift convention."""
    donors = {}
    for direction in range(8):
        ids = sorted((name for name, value in zip(day.trial_ids, day.directions)
                      if int(value) == direction), key=lambda name: int(name[6:]))
        # Write canonical ASCII JSON directly instead of calling the producer.
        key = f'[{seed},"{day.animal}","{day.day}",{direction}]'.encode("ascii")
        digest = hashlib.sha256(key).digest()
        words = [sum(digest[4 * i + j] * 256 ** j for j in range(4)) for i in range(8)]
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
        offset = int(generator.integers(1, len(ids)))
        for index, recipient in enumerate(ids):
            donors[recipient] = ids[(index + offset) % len(ids)]
    return tuple(donors[name] for name in day.trial_ids)


def key(day):
    return day.region, day.stream, day.day


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("neural values accessed before whole-roster metadata binding")


class FitMismatchTests(unittest.TestCase):
    def test_source_mapping_preserves_whole_native_rows_and_shared_regions(self):
        original = fixture()
        rng_before = np.random.get_state()
        result = core.derange_fit_lfp(original, scope="earlier", check=check)
        rng_after = np.random.get_state()
        self.assertEqual((result.scope, result.seed), ("earlier", 7007101))
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        source_mappings = {}
        for before, after, donors in zip(original, result.packs, result.lfp_source_trial_ids):
            self.assertEqual(key(before), key(after))
            self.assertEqual(before.trial_ids, after.trial_ids)
            self.assertEqual((before.role, before.view_role), (after.role, after.view_role))
            for name in ("spikes", "unit_guide", "lfp_guide", "directions", "bin_offsets"):
                self.assertIs(getattr(before, name), getattr(after, name))
            if before.role == "calibration":
                self.assertIs(before, after)
                self.assertEqual(donors, before.trial_ids)
                continue
            self.assertEqual(donors, expected_donors(before, 7007101))
            self.assertTrue(all(a != b for a, b in zip(before.trial_ids, donors)))
            self.assertEqual(set(donors), set(before.trial_ids))
            if before.day in source_mappings:
                self.assertEqual(donors, source_mappings[before.day])
            source_mappings[before.day] = donors
            indices = {name: index for index, name in enumerate(before.trial_ids)}
            for index, donor in enumerate(donors):
                np.testing.assert_array_equal(after.lfp[index], before.lfp[indices[donor]])
                self.assertEqual(after.directions[index], before.directions[indices[donor]])
            self.assertEqual(after.lfp.dtype, before.lfp.dtype)
            self.assertEqual(after.lfp.shape[1], 10 if before.region == "M1" else 15)
            with self.assertRaises(ValueError):
                after.lfp.setflags(write=True)
        self.assertEqual(set(source_mappings), set(EARLIER["Mihili"]))
        reverse_order = core.derange_fit_lfp(original[::-1], scope="earlier", check=check)
        self.assertEqual(tuple(key(day) for day in reverse_order.packs),
                         tuple(key(day) for day in original[::-1]))
        mappings = {key(day): donors for day, donors in zip(result.packs, result.lfp_source_trial_ids)}
        for day, donors in zip(reverse_order.packs, reverse_order.lfp_source_trial_ids):
            self.assertEqual(donors, mappings[key(day)])

    def test_calibration_mapping_shared_across_streams_and_numeric_trial_order(self):
        original = fixture()
        result = core.derange_fit_lfp(original, scope="calibration", check=check)
        self.assertEqual(result.seed, 7007102)
        shared = None
        for before, after, donors in zip(original, result.packs, result.lfp_source_trial_ids):
            self.assertIs(before.spikes, after.spikes)
            if before.role == "training":
                self.assertIs(before, after)
                self.assertEqual(donors, before.trial_ids)
                continue
            self.assertEqual(donors, expected_donors(before, 7007102))
            self.assertTrue(all(a != b for a, b in zip(before.trial_ids, donors)))
            if shared is not None:
                self.assertEqual(donors, shared)
            shared = donors
            first_direction = [name for name, direction in zip(before.trial_ids, before.directions)
                               if direction == 0]
            self.assertNotEqual(sorted(first_direction), sorted(first_direction, key=lambda name: int(name[6:])))
            for row, donor in enumerate(donors):
                np.testing.assert_array_equal(after.lfp[row], before.lfp[before.trial_ids.index(donor)])
        # Change storage order in every pack, preserving original identities.
        reordered = tuple(replace(day, trial_ids=day.trial_ids[::-1],
            directions=_snapshot(day.directions[::-1]), lfp=_snapshot(day.lfp[::-1]),
            spikes=_snapshot(day.spikes[::-1])) for day in original)
        again = core.derange_fit_lfp(reordered, scope="calibration", check=check)
        by_key = {key(day): dict(zip(day.trial_ids, donors))
                  for day, donors in zip(result.packs, result.lfp_source_trial_ids)}
        for day, donors in zip(again.packs, again.lfp_source_trial_ids):
            self.assertEqual(dict(zip(day.trial_ids, donors)), by_key[key(day)])

    def test_complete_original_scope_binding_support_and_cancellation_refuse(self):
        original = fixture()
        sentinel = (replace(original[0], lfp=ForbiddenArray()),) + original[1:]
        malformed = (
            sentinel[:-1],
            sentinel[:-1] + (original[0],),
            sentinel[:-1] + (replace(original[-1], target_day="20140307"),),
            sentinel[:-1] + (replace(original[-1], trial_ids=original[-1].trial_ids[::-1]),),
        )
        for packs in malformed:
            with self.assertRaises(ValueError):
                core.derange_fit_lfp(packs, scope="earlier", check=check)
        closed = sentinel[:-1] + (replace(original[-1], view_role="development_features", role="development"),)
        with self.assertRaises(PermissionError):
            core.derange_fit_lfp(closed, scope="calibration", check=check)
        with self.assertRaises(ValueError):
            core.derange_fit_lfp(sentinel, scope="held_out", check=check)
        singleton = []
        for day in original:
            if day.role == "calibration":
                directions = day.directions.copy()
                directions[np.flatnonzero(directions == 0)[1:]] = 1
                day = replace(day, directions=_snapshot(directions))
            singleton.append(day)
        with self.assertRaises(ValueError):
            core.derange_fit_lfp(tuple(singleton), scope="calibration", check=check)
        changed_calibration = original[:-1] + (replace(original[-1],
            spikes=_snapshot(original[-1].spikes + 1)),)
        with self.assertRaisesRegex(ValueError, "identical original calibration"):
            core.derange_fit_lfp(changed_calibration, scope="earlier", check=check)
        mutable = (replace(original[0], lfp=original[0].lfp.copy()),) + original[1:]
        with self.assertRaisesRegex(ValueError, "immutable adapter snapshots"):
            core.derange_fit_lfp(mutable, scope="earlier", check=check)
        calls = []
        def cancel():
            calls.append(None)
            if len(calls) == 20:
                raise RuntimeError("cancelled during original validation")
        with self.assertRaisesRegex(RuntimeError, "cancelled during original validation"):
            core.derange_fit_lfp(original, scope="earlier", check=cancel)
        self.assertEqual(len(calls), 20)


if __name__ == "__main__":
    unittest.main()
