"""The conditional cubic clock follows actual repeating-schedule arithmetic."""
import math
import unittest

from conditional_acquisition_recipe import (CYCLE_COUNT, SLOT_TICKS, SLOTS_PER_CYCLE,
                                             _time, conditional_noise_cell)
from repeating_acquisition_schedule import CYCLE_SECONDS, _shift


class ConditionalNoiseClockTests(unittest.TestCase):
    def test_exact_original_slot_support_survives_refinement_and_cycle_seams(self):
        for cycle in (0, 1, 1001, CYCLE_COUNT - 2, CYCLE_COUNT - 1):
            self.assertEqual(_shift(cycle, CYCLE_SECONDS), _shift(cycle + 1, 0.))
            for slot in (0, 1, 24, 25, 799, SLOTS_PER_CYCLE - 2, SLOTS_PER_CYCLE - 1):
                start = _shift(cycle, _time(slot * SLOT_TICKS))
                end = _shift(cycle, _time((slot + 1) * SLOT_TICKS))
                original = (start, end, start + (end - start) / 2)
                # Physical field refinement and aperture/storage subdivisions
                # query different midpoints but never choose a new noise cell.
                for left, right in ((0., 1.), (0., .5), (.5, 1.), (.125, .25)):
                    a, b = start + (end - start) * left, start + (end - start) * right
                    key = a + (b - a) / 2
                    with self.subTest(cycle=cycle, slot=slot, fraction=(left, right)):
                        self.assertEqual(conditional_noise_cell(key), original)
                for key in (math.nextafter(start, end), math.nextafter(end, start)):
                    self.assertEqual(conditional_noise_cell(key), original)
                # Actual shared represented endpoints refuse, rather than
                # silently moving a forcing key across the physical boundary.
                for key in (start, end):
                    with self.assertRaises(ValueError):
                        conditional_noise_cell(key)


if __name__ == "__main__":
    unittest.main()
