"""Four arithmetic/identity checks; execute only in coordinated Slurm.

Fabricated native arrays are calculator fixtures, not simulated scientific
recovery or evidence about any finalist. One full public 9999-draw call checks
fixed schedule integration; focused reducers test other cases without reruns.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from native_trials import NativePredictions
import primary_inference as core


def check():
    pass


def panel():
    cells = []
    directions = np.repeat(np.arange(8, dtype=np.int64), 6)
    repeat = np.tile(np.arange(6), 8)
    # Numeric suffix order crosses six/seven digits and differs from lexical
    # order. Region row orders also differ, without changing paired identities.
    ids = tuple(f"trial_{999995 + i:06d}" for i in range(48))
    calibration = tuple(f"trial_{i + 1:06d}" for i in range(24))
    for day_index, (animal, day) in enumerate(core.DAY_ORDER):
        for region_index, region in enumerate(core.REGIONS):
            bins, neurons = (10, 2) if region == "M1" else (15, 3)
            offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
            units = np.column_stack((np.arange(1, neurons + 1), np.ones(neurons, dtype=np.int64)))
            mean = (40 + 20 * day_index + 3 * directions[:, None, None]
                    + np.arange(bins)[None, :, None] + 2 * np.arange(neurons)[None, None, :])
            y = mean + np.asarray([-2, -1, 0, 0, 1, 2])[repeat, None, None]
            history = y + .2 * (1 + repeat[:, None, None] / 5)
            today = y + 1 + .1 * repeat[:, None, None]
            order = np.arange(48)[::-1] if region_index else np.roll(np.arange(48), day_index + 3)
            ordered_ids = tuple(ids[i] for i in order)
            def prediction(values):
                return NativePredictions(animal, day, region, ordered_ids,
                    directions[order], offsets, units, values[order], calibration)
            cells.append(core.HeldOutCell(animal, day, region, "held_out", ordered_ids,
                directions[order], offsets, units, y[order], prediction(history),
                prediction(today), prediction(mean)))
    return tuple(cells)


def explicit_scores(cell, occurrences):
    order = sorted(range(len(cell.trial_ids)), key=lambda index: int(cell.trial_ids[index][6:]))
    y = cell.observed[order][occurrences].astype(np.float64)
    neuron_mean = np.mean(y, axis=(0, 1), keepdims=True)
    sst = float(np.sum((y - neuron_mean)**2))
    scores = []
    for prediction in (cell.history, cell.today, cell.calibration_mean):
        p = prediction.values[order][occurrences]
        scores.append(1 - float(np.sum((y - p)**2)) / sst)
    return np.asarray(scores), sst


def explicit_aggregate(cells, occurrences):
    differences = np.empty((5, 2, 2))
    for day in range(5):
        for region in range(2):
            scores, _ = explicit_scores(cells[2 * day + region], occurrences[day])
            differences[day, region] = scores[0] - scores[1:]
    result = np.empty((2, 2))
    for region in range(2):
        for comparator in range(2):
            result[region, comparator] = (
                math.fsum(float(differences[day, region, comparator]) for day in range(2)) / 4
                + math.fsum(float(differences[day, region, comparator]) for day in range(2, 5)) / 6)
    return result


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("response/prediction values accessed before complete panel binding")


class PrimaryInferenceTests(unittest.TestCase):
    def test_duplicated_trial_oracle_and_fixed_shared_schedule_across_batches(self):
        cells = panel()
        bound = core._bind_panel(cells, check)
        statistics = core._trial_statistics(bound[0][0], check)
        # Select the first trial six times in every direction. Bootstrap
        # neuron means differ from the original means, so fixed SST is wrong.
        duplicated = np.repeat(np.arange(0, 48, 6), 6)
        counts = np.bincount(duplicated, minlength=48)[None, :]
        scores, differences, valid = core._score_multiplicities(statistics, counts)
        expected, sst = explicit_scores(cells[0], duplicated)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(scores[0], expected, rtol=0, atol=2e-14)
        np.testing.assert_allclose(differences[0], expected[0] - expected[1:], rtol=0, atol=2e-14)
        self.assertNotAlmostEqual(float((counts @ statistics.centered_square_sum)[0]), sst)

        state = np.random.get_state()
        result = core.infer_primary(cells[::-1], check=check)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertEqual(result.bootstrap_differences.shape, (9999, 2, 2))
        self.assertEqual((result.bootstrap_seed, result.bootstrap_samples, result.rng_algorithm),
                         (7007999, 9999, "PCG64"))
        np.testing.assert_array_equal(result.margins.valid_draws, [9999, 9999])
        np.testing.assert_allclose(result.observed.aggregate_differences,
            explicit_aggregate(cells, [np.arange(48)] * 5), rtol=0, atol=2e-14)
        # Independent explicit occurrence vectors preserve duplicates and
        # check two draws on each side of the production batch boundary.
        rng = np.random.Generator(np.random.PCG64(7007999))
        for draw in range(258):
            occurrences = []
            for _ in range(5):
                occurrences.append(np.concatenate([6 * direction + rng.integers(0, 6, size=6, dtype=np.int64)
                                                    for direction in range(8)]))
            if draw in (0, 1, 255, 256, 257):
                np.testing.assert_allclose(result.bootstrap_differences[draw],
                    explicit_aggregate(cells, occurrences), rtol=0, atol=3e-14)
        self.assertFalse(result.scientific_checks_certified)
        self.assertFalse(result.authorizes_opening)
        with self.assertRaises(ValueError):
            result.bootstrap_differences.setflags(write=True)

    def test_centered_tail_iut_holm_boundaries_and_unavailable_fixed_family(self):
        point = np.full((2, 2), .02)
        draws = np.broadcast_to(point, (9999, 2, 2)).copy()
        # Positive bootstrap errors, not lower percentile tails, create these
        # known p-values. Regional IUT must take the less favorable comparator.
        for region, counts in enumerate(((199, 249), (489, 399))):
            for comparator, count in enumerate(counts):
                draws[:count, region, comparator] += 1
        result = core._margin_inference(point, draws)
        np.testing.assert_array_equal(result.comparator_p, [[.02, .025], [.049, .04]])
        np.testing.assert_array_equal(result.regional_p, [.025, .049])
        np.testing.assert_array_equal(result.holm_adjusted_p, [.05, .05])
        np.testing.assert_array_equal(result.holm_rejected, [True, True])
        changed = draws.copy()
        changed[:500, 1, 0] = point[1, 0] + 1
        np.testing.assert_array_equal(core._margin_inference(point, changed).holm_rejected, [True, False])
        invalid = draws.copy()
        invalid[0, 0, 0] = np.nan
        unavailable = core._margin_inference(point, invalid)
        np.testing.assert_array_equal(unavailable.valid_draws, [9998, 9999])
        np.testing.assert_array_equal(unavailable.available, [False, True])
        self.assertTrue(np.isnan(unavailable.comparator_p[0]).all())
        self.assertTrue(np.isnan(unavailable.regional_p[0]))
        self.assertTrue(np.isnan(unavailable.holm_adjusted_p[0]))
        np.testing.assert_array_equal(unavailable.holm_input_p, [1., .049])
        self.assertAlmostEqual(unavailable.holm_adjusted_p[1], .098)
        np.testing.assert_array_equal(unavailable.holm_rejected, [False, False])
        # Equality belongs to the adverse tail: exact margin/equality gives p1.
        tied = core._margin_inference(np.full((2, 2), .01), np.full((9999, 2, 2), .01))
        np.testing.assert_array_equal(tied.comparator_p, np.ones((2, 2)))
        zero_cell = replace(panel()[0], observed=np.full_like(panel()[0].observed, 7))
        zero_statistics = core._trial_statistics(core._bind_cell(zero_cell), check)
        _, zero_difference, valid = core._score_multiplicities(zero_statistics, np.ones((1, 48)))
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(zero_difference).all())
        missing_point = point.copy()
        missing_point[0] = zero_difference[0]
        self.assertFalse(core._margin_inference(missing_point, draws).available[0])

    def test_observed_joint_day_frequency_and_all_five_strict_omission_rules(self):
        differences = np.full((5, 2, 2), .02)
        good = core._observed_gates(differences)
        np.testing.assert_array_equal(good.positive_day_counts, [[2, 2], [3, 3]])
        self.assertTrue(good.margin_pass.all())
        self.assertTrue(good.omission_pass.all())
        margin = core._observed_gates(np.full((5, 2, 2), .01))
        self.assertFalse(margin.margin_pass.any())
        zero = differences.copy()
        zero[0, 0, 0] = 0
        exact_zero = core._observed_gates(zero)
        self.assertEqual(exact_zero.positive_day_counts[0, 0], 1)
        self.assertFalse(exact_zero.omission_pass[1, 0])  # Remaining Mihili value is zero.
        self.assertTrue(exact_zero.animal_sign_pass[0])

        disjoint = differences.copy()
        disjoint[2:, 0] = [[.2, -.01], [-.01, .2], [.2, .2]]
        separate = core._observed_gates(disjoint)
        self.assertTrue(separate.animal_sign_pass[0])
        self.assertEqual(separate.positive_day_counts[1, 0], 1)
        self.assertFalse(separate.day_frequency_pass[0])  # Each comparator separately has2/3.

        influential = differences.copy()
        influential[2:, 0] = [[.04, .04], [.04, .04], [-.04, -.04]]
        failed = core._observed_gates(influential)
        self.assertTrue(failed.day_frequency_pass[0])
        self.assertTrue(failed.animal_sign_pass[0])
        self.assertFalse(failed.omission_pass[2, 0])  # Exact zero after removing one good Chewie day.
        influential[4, 0] = [-.01, -.01]
        passed = core._observed_gates(influential)
        self.assertTrue(passed.omission_pass[:, 0].all())  # No2/3 rule reapplied to remaining two days.
        for omitted in range(5):
            expected_animals = []
            for indices in ((0, 1), (2, 3, 4)):
                kept = [index for index in indices if index != omitted]
                expected_animals.append([math.fsum(float(influential[index, 0, c]) for index in kept) / len(kept)
                                         for c in range(2)])
            np.testing.assert_allclose(passed.omission_animal_differences[omitted, :, 0],
                                       expected_animals, rtol=0, atol=1e-16)
            np.testing.assert_allclose(passed.omission_aggregate_differences[omitted, 0],
                [(expected_animals[0][c] + expected_animals[1][c]) / 2 for c in range(2)], rtol=0, atol=1e-16)

    def test_complete_role_and_identity_binding_precedes_values_and_native_counts_are_required(self):
        original = panel()
        unread = tuple(replace(cell, observed=ForbiddenArray(),
            history=replace(cell.history, values=ForbiddenArray()),
            today=replace(cell.today, values=ForbiddenArray()),
            calibration_mean=replace(cell.calibration_mean, values=ForbiddenArray())) for cell in original)
        for role in ("development", "calibration", "training"):
            with self.assertRaises(PermissionError):
                core.infer_primary(unread[:-1] + (replace(unread[-1], role=role),), check=check)
        for cells in (unread[:-1], unread[:-1] + unread[:1]):
            with self.assertRaises(ValueError):
                core.infer_primary(cells, check=check)
        wrong = replace(unread[-1], today=replace(unread[-1].today, trial_ids=unread[-1].trial_ids[::-1]))
        with self.assertRaises(ValueError):
            core.infer_primary(unread[:-1] + (wrong,), check=check)
        # Locally matching predictor IDs still cannot change the PMd trial
        # support relative to simultaneously recorded M1.
        ids = unread[-1].trial_ids[:-1] + ("trial_2000001",)
        wrong = replace(unread[-1], trial_ids=ids,
            history=replace(unread[-1].history, trial_ids=ids),
            today=replace(unread[-1].today, trial_ids=ids),
            calibration_mean=replace(unread[-1].calibration_mean, trial_ids=ids))
        with self.assertRaises(ValueError):
            core.infer_primary(unread[:-1] + (wrong,), check=check)
        overlap = (unread[0].trial_ids[0],) + unread[0].history.calibration_trial_ids[1:]
        wrong = replace(unread[0], history=replace(unread[0].history, calibration_trial_ids=overlap))
        with self.assertRaises(ValueError):
            core.infer_primary((wrong,) + unread[1:], check=check)
        with self.assertRaises(ValueError):
            core.infer_primary((replace(original[0], observed=original[0].observed + .5),) + original[1:], check=check)
        def cancelled():
            raise RuntimeError("cancelled before any values")
        with self.assertRaisesRegex(RuntimeError, "cancelled before any values"):
            core.infer_primary(unread, check=cancelled)


if __name__ == "__main__":
    unittest.main()
