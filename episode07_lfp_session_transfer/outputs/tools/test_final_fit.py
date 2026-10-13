"""Two bounded final-state composition checks; run only in coordinated Slurm.

The small integer direction/time fixture has an analytic penalized ridge
solution. It checks deployment/persistence, not scientific recovery or a new
model-selection opportunity. No empirical or sealed data are used here.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import final_fit as core
from native_trials import BANDS, DayTrials, EARLIER, _snapshot
from finalist_controls import FinalistSpec, FinalistMember, resolve_finalist


def check():
    pass


def packs(animal, target, region, stream):
    offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
    cells = np.asarray([[100 + scale * (1 if not (d & bit) else -1)
                         for scale, bit in ((4, 1), (3, 2), (2, 4))]
                        + [100 + (1 if b % 2 == 0 else -1)]
                        for d in range(8) for b in range(len(offsets))], dtype=np.int64)
    cells = cells.reshape(8, len(offsets), 4)
    dates = (EARLIER[animal] if stream == "history_assisted" else ()) + (target,)
    days = []
    for index, date in enumerate(dates):
        native_index = EARLIER[animal].index(date) if date in EARLIER[animal] else 3
        count = 3 if date == target else 15
        directions = np.repeat(np.arange(8, dtype=np.int64), count)
        days.append(DayTrials(animal, date, target, region, stream, "fit",
            "calibration" if date == target else "training",
            tuple(f"trial_{i + 1:06d}" for i in range(len(directions))),
            _snapshot(directions), _snapshot(offsets),
            _snapshot(np.zeros((len(directions), len(offsets), 9), dtype=np.float32)),
            _snapshot(np.asarray([(native_index + 1, low, high) for low, high in BANDS])),
            _snapshot(cells[directions]),
            # Historical native labels are deliberately unrelated to target labels.
            _snapshot(np.asarray([(100 * native_index + n + 1, 1) for n in range(4)]))))
    calibration = days[-1]
    held_directions = np.repeat(np.arange(8, dtype=np.int64), 6)
    held = replace(calibration, view_role="held_out_features", role="held_out",
        trial_ids=tuple(f"trial_{1001 + i:06d}" for i in range(48)),
        directions=_snapshot(held_directions),
        lfp=_snapshot(np.zeros((48, len(offsets), 9), dtype=np.float32)),
        spikes=None, unit_guide=None)
    return tuple(days), held, cells


def ridge_oracle(cells, penalty):
    # Equal cell mass, unpenalized intercept and all cell indicators except the
    # first. Solve the scalar first-order equations, independently of regression
    # or PCA implementation. Full rank four loses no native response dimension.
    flat = cells.reshape(-1, 4).astype(np.float64)
    count = len(flat)
    reference = (flat[0] + count**2 * penalty * flat.mean(axis=0)) / (1 + count**2 * penalty)
    fitted = reference + (flat - reference) / (1 + count * penalty)
    fitted[0] = reference
    return fitted.reshape(cells.shape)


class FinalFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = tuple(resolve_finalist("starting_00", stream=stream) for stream in core.STREAMS)
        cells, cls.inputs = [], {}
        for spec in cls.specs:
            for animal, target, region in core.PANEL_KEYS:
                fit_days, held, oracle_cells = packs(animal, target, region, spec.stream)
                cell = core.fit_final_cell(spec, fit_days, check=check)
                cells.append(cell)
                cls.inputs[(spec.stream, animal, target, region)] = (held, oracle_cells)
        cls.pair = core.assemble_final_pair(cls.specs, tuple(reversed(cells)), check=check)

    def test_complete_pair_round_trip_preserves_analytic_predictions_and_fixed_native_ensemble(self):
        expected_keys = tuple((stream, *key) for stream in core.STREAMS for key in core.PANEL_KEYS)
        self.assertEqual(tuple((c.stream, c.animal, c.target_day, c.region) for c in self.pair.cells),
                         expected_keys)
        before = []
        for cell in self.pair.cells:
            held, cells = self.inputs[(cell.stream, cell.animal, cell.target_day, cell.region)]
            prediction, baseline = core.predict_final_cell(cell, held, check=check)
            oracle = ridge_oracle(cells, .001)[held.directions]
            np.testing.assert_allclose(prediction.values, oracle, rtol=0, atol=2e-9)
            np.testing.assert_array_equal(baseline.values, cells[held.directions])
            self.assertEqual(prediction.calibration_trial_ids, cell.calibration.calibration_trial_ids)
            before.append(prediction.values)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen_final_pair.pkl"
            with patch.object(core, "FINAL_PAIR_PATH", path):
                core.save_final_pair(path, self.pair, check=check)
                self.assertFalse(path.with_name(path.name + ".partial").exists())
                with patch.object(core, "fit_primary_model", side_effect=AssertionError("unexpected refit")), \
                     patch.object(core, "fit_weighted_primary_model", side_effect=AssertionError("unexpected weight refit")):
                    loaded = core.load_final_pair(path, self.specs, check=check)
                    for cell, expected in zip(loaded.cells, before):
                        held, _ = self.inputs[(cell.stream, cell.animal, cell.target_day, cell.region)]
                        actual, baseline = core.predict_final_cell(cell, held, check=check)
                        np.testing.assert_array_equal(actual.values, expected)
                        np.testing.assert_array_equal(actual.unit_guide, baseline.unit_guide)
                        with self.assertRaises(ValueError):
                            cell.members[0].fitted.regression.coefficients.setflags(write=True)
                with self.assertRaises(FileExistsError):
                    core.save_final_pair(path, self.pair, check=check)

        # Pure composition fixture: the resolver supplies one declared two-member
        # ensemble for this subcase only. Both frozen member fits have identical
        # training bindings; their known native intercept shifts differ. No
        # production candidate is selected or its declaration modified.
        cell = self.pair.cells[0]
        original = cell.members[0].fitted
        declarations = tuple(FinalistMember(name, original.recipe, None, weight)
                             for name, weight in (("fixture_a", .25), ("fixture_b", .75)))
        spec = FinalistSpec("fixture_ensemble", cell.stream, declarations)
        target_basis = original.population.bases[-1]
        members = []
        for declaration, shift in zip(declarations, (2., 6.)):
            output_shift = np.full(4, shift) @ target_basis.components
            regression = replace(original.regression,
                output_mean=_snapshot(original.regression.output_mean + output_shift))
            members.append(core.FrozenFinalMember(declaration.candidate_id,
                                                  replace(original, regression=regression)))
        ensemble = replace(cell, finalist=spec, members=tuple(members))
        held, _ = self.inputs[(cell.stream, cell.animal, cell.target_day, cell.region)]
        with patch.object(core, "resolve_finalist", return_value=spec), \
             patch.object(core, "fit_primary_model", side_effect=AssertionError("unexpected refit")):
            prediction, _ = core.predict_final_cell(ensemble, held, check=check)
        np.testing.assert_allclose(prediction.values, before[0] + 5., rtol=0, atol=2e-12)

    def test_partial_mismatched_unfitted_and_wrong_phase_states_are_refused(self):
        cell = self.pair.cells[0]
        held, _ = self.inputs[(cell.stream, cell.animal, cell.target_day, cell.region)]
        for cells in (self.pair.cells[:-1], self.pair.cells + (cell,)):
            with self.assertRaises(ValueError):
                core.assemble_final_pair(self.specs, cells, check=check)
        primary = cell.members[0].fitted
        for bad in (replace(cell, members=()),
                    replace(cell, members=(replace(cell.members[0], fitted=None),)),
                    replace(cell, members=(replace(cell.members[0], fitted=replace(primary,
                        features=replace(primary.features, stream="today_only"))),)),
                    replace(cell, members=(replace(cell.members[0], fitted=replace(primary,
                        regression=replace(primary.regression, coefficients=np.zeros((1, 1))))),)),
                    replace(cell, calibration=replace(cell.calibration,
                        means=cell.calibration.means + 1))):
            with self.assertRaises((ValueError, TypeError)):
                core.assemble_final_pair(self.specs, (bad,) + self.pair.cells[1:], check=check)
        # A self-consistent target identity change in just one stream/region
        # must still be caught by the complete pair's shared-calibration binding.
        shifted_ids = tuple(f"trial_{5001 + i:06d}" for i in range(len(cell.calibration.calibration_trial_ids)))
        f = replace(primary.features, fit_trial_ids=primary.features.fit_trial_ids[:-1]
                    + ((cell.target_day, shifted_ids),))
        p = replace(primary.population, bases=primary.population.bases[:-1]
                    + (replace(primary.population.bases[-1], trial_ids=shifted_ids),))
        d = replace(primary.design, day_means=primary.design.day_means[:-1]
                    + (replace(primary.design.day_means[-1], trial_ids=shifted_ids),))
        changed = replace(cell, calibration=replace(cell.calibration, calibration_trial_ids=shifted_ids),
            members=(replace(cell.members[0], fitted=replace(primary, features=f, population=p, design=d)),))
        with self.assertRaisesRegex(ValueError, "share calibration"):
            core.assemble_final_pair(self.specs, (changed,) + self.pair.cells[1:], check=check)
        with patch.object(core, "predict_primary_held_out", side_effect=AssertionError("unexpected prediction")):
            for bad in (replace(held, view_role="development_features", role="development"),
                        replace(held, spikes=np.zeros((48, 10, 4))),
                        replace(held, stream="today_only")):
                with self.assertRaises(PermissionError):
                    core.predict_final_cell(cell, bad, check=check)
        with patch.object(core, "fit_primary_model", side_effect=AssertionError("unexpected fit")):
            with self.assertRaises(PermissionError):
                core.fit_final_cell(self.specs[0], (held,), check=check)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen_final_pair.pkl"
            with patch.object(core, "FINAL_PAIR_PATH", path):
                with self.assertRaises(ValueError):
                    core.save_final_pair(path.with_name("external.pkl"), self.pair, check=check)
                core.save_final_pair(path, self.pair, check=check)
                wrong = (resolve_finalist("starting_01", stream=core.STREAMS[0]), self.specs[1])
                with self.assertRaises(ValueError):
                    core.load_final_pair(path, wrong, check=check)
                path.unlink()
                partial = path.with_name(path.name + ".partial")
                partial.write_bytes(b"interrupted fixture")
                with self.assertRaises(FileExistsError):
                    core.save_final_pair(path, self.pair, check=check)
                self.assertEqual(partial.read_bytes(), b"interrupted fixture")
        def cancelled():
            raise RuntimeError("fixture cancelled")
        with self.assertRaisesRegex(RuntimeError, "fixture cancelled"):
            core.predict_final_cell(cell, held, check=cancelled)


if __name__ == "__main__":
    unittest.main()
