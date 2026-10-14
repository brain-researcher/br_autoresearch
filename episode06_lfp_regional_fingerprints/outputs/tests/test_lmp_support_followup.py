"""Focused EP06 support/role arithmetic checks; run only in coordinated Slurm."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as bp
import run_lmp_support_followup as run
from test_bandwise_families import cca_fixture
from test_elapsed_time_baseline import metadata


def check():
    pass


class FollowupTests(unittest.TestCase):
    def test_changed_unit_population_context_timing_and_scoring_remain_bound(self):
        train, counts, u, v = cca_fixture()
        held, observed, hu, hv = cca_fixture(evaluation=True)
        def add_unit(features, response, x, z):
            design = replace(features.design, axis_ids=features.design.axis_ids+("third",))
            response = replace(response, design=design,
                values=np.concatenate((response.values, (100+2*x+z)[:, :, None]), axis=2))
            return replace(features, design=design), response
        train, counts = add_unit(train, counts, u, v)
        held, observed = add_unit(held, observed, hu, hv)
        fit = bp.fit_bandwise(train, counts, recipe=run.RECIPE, check=check)
        original = {"fit": fit, "predictions": bp.predict_bandwise(fit, held, check=check)}
        tm = replace(metadata(train, 100+u[:, 0]), native_unit_guide=((1., 1.), (2., 1.), (3., 1.)))
        em = replace(metadata(held, 100+hu[:, 0]), native_unit_guide=tm.native_unit_guide)
        plan = {"id": "disjoint_electrodes_0", "indices": [0, 2], "support": "disjoint_electrodes", "draw": 0}
        cell = run.fit_cell(run.SESSIONS[0], plan, original, (train, counts), held, tm, em, check)
        self.assertNotIn("unavailable", cell)
        self.assertEqual(cell["context"].training_design.axis_ids, ("neuron_a", "third"))
        self.assertEqual(fit.training_design.axis_ids, ("neuron_a", "neuron_b", "third"))
        scored = run.score_cell(cell, observed, run.mismatch_indices(held.design, [3], number=3), check)
        self.assertNotIn("unavailable", scored)
        self.assertEqual(len(scored["methods"]["lmp"]["mismatch_scores"]), 3)
        self.assertIsInstance(scored["elapsed_score"], float)

    def test_one_band_map_reproduces_original_frozen_lmp_and_no_outcome_input(self):
        train, counts, _, _ = cca_fixture()
        held, _, _, _ = cca_fixture(evaluation=True)
        fitted = bp.fit_bandwise(train, counts, recipe=run.RECIPE, check=check)
        original = bp.predict_bandwise(fitted, held, check=check).predictions[0].values
        target = bp.transform_observations(fitted, counts, region="M1", check=check).values
        direct = run.fit_map(train.values[:, :, :, 0].reshape(-1, 15), target.reshape(-1, 2), check)
        actual = run.predict_map(direct, held.values[:, :, :, 0].reshape(-1, 15), check).reshape(original.shape)
        np.testing.assert_allclose(actual, original, rtol=1e-12, atol=1e-12)
        # HF projection freezes only training state despite changed held inputs.
        x = train.values[:, :, :, 0].reshape(-1, 15)
        hf = train.values[:, :, :, 7:9].reshape(-1, 30)
        a, _, state = run.remove_hf(x, hf, x, hf)
        b, _, changed = run.remove_hf(x, hf, x+1000, hf-30)
        np.testing.assert_array_equal(a, b)
        for key in state:
            np.testing.assert_array_equal(state[key], changed[key])

    def test_global_count_matching_disjoint_support_halves_and_mismatch(self):
        metadata = {s.filename: SimpleNamespace(selected_electrode_ids=(1., 2.),
            native_unit_guide=((1., 1.), (1., 2.), (2., 1.), (3., 1.), (4., 1.))) for s in run.SESSIONS}
        plan = run.support_plan(metadata)
        self.assertEqual(plan["matched_unit_count"], 2)
        self.assertEqual(plan, run.support_plan(metadata))
        for cells in plan["plans"].values():
            self.assertEqual(len(cells), 7)
            for cell in cells[1:]:
                self.assertEqual(len(cell["indices"]), 2)
                pool = {0, 1, 2} if cell["support"] == "on_electrodes" else {3, 4}
                self.assertTrue(set(cell["indices"]) <= pool)
        train, _, _, _ = cca_fixture()
        halves = run.half_rows(train.design)
        self.assertFalse(set(halves[0]) & set(halves[1]))
        self.assertEqual(sorted(np.concatenate(halves)), list(range(320)))
        for mapping in run.mismatch_indices(train.design, [1, 2], number=7):
            self.assertEqual(sorted(mapping), list(range(32)))
            self.assertTrue(np.all(mapping != np.arange(32)))
            self.assertEqual(tuple(train.design.trial_directions[i] for i in mapping), train.design.trial_directions)
        metadata[run.SESSIONS[0].filename].native_unit_guide = ((1., 1.), (3., 1.))
        unavailable = run.support_plan(metadata)
        self.assertFalse(unavailable["matched_support_available"])
        self.assertTrue(all(len(v) == 1 for v in unavailable["plans"].values()))

    def test_all_dates_and_equal_animal_weight_unavailable_not_dropped(self):
        rows = []
        for spec in run.SESSIONS:
            value = 1. if spec.animal == "Mihili" else 3.
            for support in ("all_units", "on_electrodes", "disjoint_electrodes"):
                for draw in range(1 if support == "all_units" else run.DRAWS):
                    rows.append({"session": spec.filename, "animal": spec.animal,
                        "plan": {"support": support, "draw": draw},
                        "methods": {m: {"score": value, "minus_elapsed": value, "minus_mismatch": value} for m in run.METHODS}})
        result = run.summary(rows)
        self.assertEqual(result["disjoint_electrodes/lmp"]["equal_animal_mean"]["score"], 2.)
        self.assertEqual(len(result["disjoint_electrodes/lmp"]["omission_equal_animal_means"]), 7)
        rows[-1]["unavailable"] = "rank unavailable"
        result = run.summary(rows)
        self.assertFalse(result["disjoint_electrodes/lmp"]["complete"])
        self.assertIsNone(result["disjoint_electrodes/lmp"]["equal_animal_mean"])
        self.assertEqual(len(result["disjoint_electrodes/lmp"]["dates"]), 7)
        self.assertTrue(result["on_electrodes/lmp"]["complete"])


if __name__ == "__main__":
    unittest.main()
