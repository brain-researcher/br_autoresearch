"""Independent scientific checks for actual one-band acquired LMP models."""
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import lmp_acquired_reader as reader
import lmp_correspondence as selector
import lmp_correspondence_experiment as experiment
import lmp_selected_rank as selected_rank
import development_lmp_correspondence as runner
from capacity_training import fit_capacity
from initializer_target_label_control import within_direction_map


def initial_design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != fold), np.flatnonzero(slots == fold)) for fold in range(4)]
    return labels, folds


class LMPOnlyTest(unittest.TestCase):
    def test_reader_selects_only_native_lmp_and_accounts_actual_one_band_dtype(self):
        guide = np.array([[channel, lo, hi] for channel in (1, 2)
                          for lo, hi in ((0, 0), (100, 200), (200, 400))], dtype=float)
        units = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
        lfp = np.full((6, 10), np.nan, dtype=np.float32)
        lfp[0], lfp[3] = np.arange(1, 11), np.arange(11, 21)
        spikes = np.vstack((np.arange(10), np.full(10, .5), 2*np.arange(10))).astype(np.float64)
        selections = []

        class NeuralNode:
            shape = lfp.shape
            def __getitem__(self, key):
                physical_rows, interval = key
                positions = np.asarray(physical_rows).tolist()
                if not set(positions) <= {0, 3}:
                    raise AssertionError("High-band numeric values must never be read")
                selections.append(positions)
                return lfp[key]

        class Group:
            pass

        group = Group()
        class Handle:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter(("trial_data",))
            def __getitem__(self, _): return group

        fields = {"M1_lfp_guide": guide.T, "M1_unit_guide": units.T,
                  "M1_lfp": NeuralNode(), "M1_spikes": spikes}
        filename = "Mihili_CO_20140217.mat"
        source = {"filename": filename, "session_role": "development", "common_physical_channels": [1, 2]}
        role = {"filename": filename, "session_role": "development", "evaluation_trials": [1],
                "nested_training_sets": {"96": {"training_trials": [0]}}}
        records = {0: {"window_start_zero_based": 0, "window_end_exclusive": 10,
                       "array_shapes": {"M1_lfp": [10, 6], "M1_spikes": [10, 3]}}}
        report = defaultdict(int)
        exposure = reader.Exposure(report)
        with patch.object(reader.h5py, "File", return_value=Handle()), \
                patch.object(reader.h5py, "Group", Group), \
                patch.object(reader, "mclass", return_value="struct"), \
                patch.object(reader, "field", side_effect=lambda handle, group, name, trial: fields[name]):
            x, y, reference = reader.read_trials(Path(filename), source, role, records,
                                                [0], [2, 1], lambda: None, exposure)
            reader.read_trials(Path(filename), source, role, records, [0], [2], lambda: None,
                               exposure, reference)
        np.testing.assert_array_equal(x, [[15.5, 5.5]])
        np.testing.assert_array_equal(y, [[4.5, .5, 9.]])
        self.assertEqual(selections, [[0, 3], [3]])
        # First:20 float32 LMP +30 float64 target elements. Second repeats one
        # LMP channel and the same targets; source-read bytes repeat, unique do not.
        self.assertEqual(report["logical_neural_bytes"], 20*4+30*8+10*4+30*8)
        self.assertEqual(report["unique_logical_neural_bytes"], 20*4+30*8)
        self.assertEqual(report["fitting_trial_read_calls"], 2)

    def test_scalar_gain_matches_native_true_target_formula_and_original_maps(self):
        rng = np.random.default_rng(58001)
        labels, _ = initial_design()
        x, y = rng.normal(size=(32, 5)), rng.normal(size=(32, 3))
        ids, channels = np.arange(32)*13+100, [1, 2, 3, 4, 5]
        dx = x-np.stack([x[labels == d].mean(axis=0) for d in range(8)])[labels]
        dxsd = dx.std(axis=0)
        dxsd[dxsd == 0] = 1
        xx = (dx-dx.mean(axis=0))/dxsd
        ysd = y.std(axis=0)
        ysd[ysd == 0] = 1
        yy = (y-np.stack([y[labels == d].mean(axis=0) for d in range(8)])[labels])/ysd
        yy -= yy.mean(axis=0)
        gram = np.mean(xx**2, axis=0)
        for replicate in (None, 0, 4):
            if replicate is None:
                mapping, expected_maps = np.arange(32), []
            else:
                mapping, expected_maps = within_direction_map(ids, labels, "example", "outer", replicate)
            cross = xx.T@yy[mapping]/32
            expected = np.sum(cross**2, axis=1)/((1+gram)*3)
            order, scores, maps = selector.rank_view(x, y, labels, ids, channels, "example", "outer", replicate)
            np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(order, np.lexsort((channels, -expected)))
            self.assertEqual(maps, expected_maps)

    def test_control_maps_accept_identity_without_target_or_channel_mutation(self):
        rng = np.random.default_rng(58002)
        labels, _ = initial_design()
        x, y = rng.normal(size=(32, 4)), rng.normal(size=(32, 2))
        old_x, old_y = x.copy(), y.copy()
        ids, channels = np.arange(32)*7, [1, 2, 3, 4]
        class Identity:
            def seed(self, literal, version): self.version = version
            def shuffle(self, values): pass
        true = selector.rank_view(x, y, labels, ids, channels, "example", "outer", None)
        with patch("initializer_target_label_control.random.Random", side_effect=Identity):
            control = selector.rank_view(x, y, labels, ids, channels, "example", "outer", 3)
        np.testing.assert_array_equal(control[0], true[0])
        np.testing.assert_array_equal(control[1], true[1])
        self.assertEqual(len(control[2]), 8)
        for direction in control[2]:
            self.assertEqual(direction["recipient_trial_ids"], direction["donor_trial_ids"])
        np.testing.assert_array_equal(x, old_x)
        np.testing.assert_array_equal(y, old_y)

    def test_actual_E_indexing_and_legal_rank_match_unpadded_decoder(self):
        rng = np.random.default_rng(58003)
        labels, folds = initial_design()
        x, y = rng.normal(size=(32, 96)), rng.normal(size=(32, 10))
        order = np.arange(96)[::-1]
        fitted, columns = selected_rank.fit_selected_rank(x, y, labels, folds,
            [order.copy() for _ in folds], order, 2, defaultdict(int))
        np.testing.assert_array_equal(columns, [95, 94])
        self.assertEqual(fitted.coefficients.shape, (2, 10))
        self.assertEqual(fitted.legal_ranks, (1, 2, "full"))
        self.assertEqual(fitted.candidate_count, 15)
        direct = fit_capacity(x[:, [95, 94]], y, labels, folds, "reduced_rank", defaultdict(int))
        self.assertEqual((fitted.rank, fitted.alpha), (direct.rank, direct.alpha))
        np.testing.assert_allclose(fitted.coefficients, direct.coefficients, rtol=1e-12, atol=1e-13)

    def test_each_inner_view_excludes_its_validation_X_and_Y_from_all_orders(self):
        rng = np.random.default_rng(58005)
        labels, folds = initial_design()
        x, y = rng.normal(size=(32, 96)), rng.normal(size=(32, 10))
        ids, channels = np.arange(32)*11+100, list(range(1, 97))
        before, records = selector.order_bank(x, y, labels, ids, channels, folds, "example", defaultdict(int))
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += rng.normal(size=(8, 96))*1e5
        changed_y[folds[0][1]] -= rng.normal(size=(8, 10))*1e5
        after, changed = selector.order_bank(changed_x, changed_y, labels, ids, channels, folds,
                                             "example", defaultdict(int))
        for seed in selector.BRANCHES:
            np.testing.assert_array_equal(before[seed]["0"], after[seed]["0"])
            original = next(row for row in records if row["mc_seed"] == seed and row["view"] == "0")
            other = next(row for row in changed if row["mc_seed"] == seed and row["view"] == "0")
            self.assertEqual(original, other)

    def test_all_30_orders_precede_future_reads_and_all_heads_use_true_Y(self):
        rng = np.random.default_rng(58004)
        filename = "Mihili_CO_20140217.mat"
        pools = [list(range(12*d, 12*d+12)) for d in range(8)]
        ids = [trial for pool in pools for trial in pool[:4]]
        labels, folds = initial_design()
        role = {"filename": filename, "animal": "Mihili", "fitting_pool_by_direction": pools,
                "nested_training_sets": {"32": {"training_trials": ids, "inner_folds": [
                    {"fitting_trials": [ids[index] for index in fit],
                     "validation_trials": [ids[index] for index in valid]} for fit, valid in folds]}}}
        records = {trial: {"direction_index": direction} for direction, pool in enumerate(pools) for trial in pool}
        source = {"common_physical_channels": list(range(1, 97))}
        x, y = rng.normal(size=(32, 96)), rng.normal(size=(32, 10))
        true_y = y.copy()
        progress = defaultdict(int)
        calls = []
        case = self

        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                self.channels = channels
            def initialize_selected(self, xx, yy, channels, guide):
                np.testing.assert_array_equal(yy, true_y)
                case.assertEqual(xx.shape, (32, 96))

        def model():
            progress["outer_training_refits"] += 1
            return SimpleNamespace(number=progress["outer_training_refits"])
        def full_fit(xx, yy, dd, ff, method, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("full_true_target")
            return model()
        def selected_fit(xx, yy, dd, ff, inner, outer, count, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("scarce_true_target")
            return model(), np.asarray(outer[:count])
        def later(trajectory, total, records, pp, make_null=False):
            case.assertEqual(progress["selector_orderings"], 30)
            calls.append("future_acquisition")
            return model(), object() if make_null else None

        with patch.object(experiment, "read_trials", return_value=(x, y, object())) as reads, \
                patch.object(experiment, "Trajectory", Trajectory), \
                patch.object(experiment, "fit_capacity", side_effect=full_fit), \
                patch.object(experiment, "fit_selected_rank", side_effect=selected_fit), \
                patch.object(experiment, "later_fit", side_effect=later):
            models, nulls, guide, selections = experiment.fit_target(Path(filename), source, role, records,
                progress, lambda: None, object())
        self.assertEqual(len(selections), 30)
        self.assertEqual(progress["direction_permutation_maps"], 200)
        self.assertEqual(progress["outer_training_refits"], 95)
        self.assertEqual(len(models), 120)
        self.assertEqual(len({id(item[-1]) for item in models}), 95)
        self.assertEqual(calls.count("scarce_true_target"), 18)
        self.assertEqual(calls.count("future_acquisition"), 76)
        self.assertEqual(reads.call_count, 1)
        np.testing.assert_array_equal(y, true_y)

    def test_runner_rejects_94_models_before_any_evaluation_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                reference_report=root/"closed_initializer.json", acquired_reference=root/"closed_acquired.json",
                output_dir=root/"failed_attempt", max_seconds=270)
            def incomplete(path, source, role, records, progress, check, exposure):
                progress["outer_training_refits"] += 94
                progress["selector_orderings"] += 30
                return [], {}, None, []
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "selected_metadata", return_value=({}, {}, {})), \
                    patch.object(runner, "fit_target", side_effect=incomplete), \
                    patch.object(runner, "read_trials") as evaluation:
                with self.assertRaisesRegex(ArithmeticError, "All95"):
                    runner.main()
            evaluation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
