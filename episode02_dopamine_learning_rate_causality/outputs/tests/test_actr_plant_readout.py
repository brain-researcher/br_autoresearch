"""Small event-tape fixtures; no recurrent training or cohort simulation."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from actr_plant_readout import (
    FrozenPlantTrial, FrozenSeedPath, MonteCarloSettings, PlantReadoutError,
    conditional_plant_readouts,
)


def paths():
    return tuple(FrozenSeedPath(seed, (
        FrozenPlantTrial("s1", np.zeros(1000), 900, np.ones(1000), 50.0, True),
    )) for seed in (1, 2, 3, 4, 5))


def settings(**kwargs):
    base = MonteCarloSettings((4, 16), .2, 4.1, 80)
    return replace(base, **kwargs)


def run(seed_paths=None, config=None, check=lambda: None):
    return conditional_plant_readouts(mouse_id=1, session_ids=("s1",),
        seed_paths=paths() if seed_paths is None else seed_paths, readout_seed=729,
        settings=settings() if config is None else config, check=check)


class AlternatingPlant:
    """An exact half-success tape for inspecting numerical-bound arithmetic."""
    def __init__(self):
        self.draws = {}
        self.inputs_frozen = True

    def __call__(self, activity, rng, *, reward, scale):
        self.inputs_frozen &= not activity.flags.writeable and not reward.flags.writeable
        count = self.draws.get(rng, 0)
        self.draws[rng] = count + 1
        # R=900: left boundary R-750=150 is included; R itself is excluded.
        return (np.array([150]) if count % 2 == 0 else np.array([900])), None


class ActrPlantReadoutTests(unittest.TestCase):
    def test_actual_source_plant_zero_drive_with_explicit_reward_has_no_licks(self):
        # Five calls of the real port, one 1000-tick activity per seed; no RNN
        # or learning simulation. Explicit reward disables the source's
        # default background drive, so zero input can never leave state zero.
        result = run(config=settings(draws_per_trial=(1,), maximum_total_plant_draws=5))
        self.assertEqual(result.total_plant_draws, 5)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.readouts)
        self.assertTrue(all(estimate.probability == 0.0 for estimate in result.estimates))

    def test_five_paths_and_stagewise_error_allocation(self):
        plant = AlternatingPlant()
        with patch("actr_plant_readout.behavioral_plant", side_effect=plant):
            result = run()
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.total_plant_draws, 80)
        self.assertEqual(len(result.readouts), 5)
        self.assertAlmostEqual(result.per_interval_failure_probability, .2 / (5 * 1 * 2))
        radius = math.sqrt(math.log(100) / 32)
        for estimate, readout in zip(result.estimates, result.readouts):
            self.assertEqual(estimate.draws_per_trial, 16)
            self.assertEqual(estimate.probability, .5)
            self.assertAlmostEqual(estimate.probability_lower, .5 - radius)
            self.assertAlmostEqual(estimate.probability_upper, .5 + radius)
            self.assertEqual(readout.probabilities, (.5,))
            self.assertEqual(readout.model_trial_counts, (1,))
        self.assertTrue(plant.inputs_frozen)

    def test_zero_and_one_draw_frequencies_stay_unresolved(self):
        for licks, expected in ((np.array([]), 0.0), (np.array([899]), 1.0)):
            with self.subTest(expected=expected), patch("actr_plant_readout.behavioral_plant", return_value=(licks, None)):
                result = run()
            self.assertEqual(result.status, "unresolved")
            self.assertIsNone(result.readouts)
            self.assertTrue(all(row.probability == expected for row in result.estimates))
            self.assertTrue(all(math.isinf(row.logit_width) for row in result.estimates))

    def test_draw_cap_preserves_last_complete_stage_without_subset_handoff(self):
        plant = AlternatingPlant()
        with patch("actr_plant_readout.behavioral_plant", side_effect=plant):
            result = run(config=settings(maximum_total_plant_draws=79))
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.reason, "draw_limit_before_next_complete_stage")
        self.assertEqual(result.total_plant_draws, 20)
        self.assertIsNone(result.readouts)
        self.assertTrue(all(row.draws_per_trial == 4 for row in result.estimates))

    def test_stochastic_replay_has_separate_rng_and_immutable_inputs(self):
        original_paths = paths()
        before = original_paths[0].trials[0].activity.copy()
        global_state = np.random.get_state()
        def bernoulli(activity, rng, **kwargs):
            return (np.array([899]) if rng.random() < .5 else np.array([])), None
        with patch("actr_plant_readout.behavioral_plant", side_effect=bernoulli):
            first = run(seed_paths=original_paths)
            second = run(seed_paths=original_paths)
        self.assertEqual(first, second)
        after = np.random.get_state()
        self.assertEqual(global_state[0], after[0])
        np.testing.assert_array_equal(global_state[1], after[1])
        self.assertEqual(global_state[2:], after[2:])
        np.testing.assert_array_equal(original_paths[0].trials[0].activity, before)

    def test_incomplete_or_undefined_scientific_inputs_fail_before_sampling(self):
        original = paths()
        bad_trial = replace(original[0].trials[0], reward_tick=700)
        missing = replace(original[0].trials[0], eligible=False)
        for invalid in (original[:-1], original[:-1] + (original[0],),
                        (replace(original[0], trials=(bad_trial,)),) + original[1:],
                        (replace(original[0], trials=(missing,)),) + original[1:]):
            with patch("actr_plant_readout.behavioral_plant") as plant:
                with self.assertRaises(PlantReadoutError):
                    run(seed_paths=invalid)
                plant.assert_not_called()
        with self.assertRaises(PlantReadoutError):
            run(config=settings(draws_per_trial=(4, 4)))

    def test_resource_stop_never_returns_a_partial_fit_ready_packet(self):
        def stop():
            raise RuntimeError("allocation ended")
        with patch("actr_plant_readout.behavioral_plant") as plant:
            with self.assertRaisesRegex(RuntimeError, "allocation ended"):
                run(check=stop)
            plant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
