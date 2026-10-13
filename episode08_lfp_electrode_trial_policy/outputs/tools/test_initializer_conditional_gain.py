"""Scientific checks for initializer-only greedy conditional selection."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_conditional_gain as selector
import initializer_conditional_experiment as experiment
import development_initializer_conditional as runner
from initializer_target_label_control import within_direction_map
from residual_selection_training import selector_coordinates, selector_order


def design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(4)]
    return labels, folds


def direct_greedy(x, y, channels, maximum):
    """Independent objective: rebuild A by direct solve after every chosen block."""
    selected, gains = [], []
    for _ in range(maximum):
        columns = [3*e+b for e in selected for b in range(3)]
        retained = x[:, columns]
        a = np.linalg.solve(np.eye(len(x))+retained@retained.T/len(x), np.eye(len(x)))
        available = [e for e in range(len(channels)) if e not in selected]
        scores = {}
        for e in available:
            block = x[:, 3*e:3*e+3]
            gram, cross = block.T@a@block/len(x), block.T@a@y/len(x)
            scores[e] = np.trace(cross.T@np.linalg.solve(np.eye(3)+gram, cross))/y.shape[1]
        chosen = min(available, key=lambda e: (-scores[e], channels[e]))
        selected.append(chosen)
        gains.append(scores[chosen])
    return np.asarray(selected), np.asarray(gains)


class InitializerConditionalTest(unittest.TestCase):
    def test_rank_three_updates_match_direct_inverse_after_each_choice(self):
        rng = np.random.default_rng(82001)
        x, y = rng.normal(size=(24, 18)), rng.normal(size=(24, 5))
        x[:, 3:6] += .8*x[:, :3]
        channels = [2, 7, 11, 19, 23, 31]
        original_x, original_y = x.copy(), y.copy()
        order, gains, _ = selector.greedy_order(x, y, channels, maximum=6)
        expected, expected_gains = direct_greedy(x, y, channels, 6)
        np.testing.assert_array_equal(order, expected)
        np.testing.assert_allclose(gains, expected_gains, rtol=2e-11, atol=2e-13)
        np.testing.assert_array_equal(x, original_x)
        np.testing.assert_array_equal(y, original_y)

    def test_zero_gain_ties_do_not_stop_or_reselect_channels(self):
        channels = list(range(1, 97))
        order, gains, ties = selector.greedy_order(np.zeros((24, 288)), np.zeros((24, 4)), channels)
        np.testing.assert_array_equal(order, np.arange(32))
        np.testing.assert_array_equal(gains, np.zeros(32))
        self.assertEqual(len(set(order.tolist())), 32)
        self.assertEqual(list(ties), list(range(96, 64, -1)))

    def test_identity_control_replays_inherited_conditional_order(self):
        rng = np.random.default_rng(82002)
        labels, _ = design()
        x, y = rng.normal(size=(32, 24)), rng.normal(size=(32, 6))
        ids, channels = np.arange(32)*13+100, list(range(1, 9))
        expected = selector_order(x, y, labels, "residual_conditional", 8, "unused")
        true, _ = selector.rank_view(x, y, labels, ids, channels, "example", "outer", maximum=8)
        with patch.object(selector, "within_direction_map", return_value=(np.arange(32), [])):
            control, _ = selector.rank_view(x, y, labels, ids, channels, "example", "outer", 2, maximum=8)
        np.testing.assert_array_equal(true, expected)
        np.testing.assert_array_equal(control, expected)

    def test_original_maps_and_independent_control_geometry_match_direct_objective(self):
        rng = np.random.default_rng(82003)
        labels, _ = design()
        x, y = rng.normal(size=(32, 18)), rng.normal(size=(32, 5))
        ids, channels = np.arange(32)*7+20, list(range(1, 7))
        xx, yy = selector_coordinates(x, y, labels)
        old_x, old_y = x.copy(), y.copy()
        control_orders = []
        for seed in (0, 2, 4):
            mapping, maps = within_direction_map(ids, labels, "example", "outer", seed)
            order, details = selector.rank_view(x, y, labels, ids, channels, "example", "outer", seed, maximum=6)
            expected, expected_gains = direct_greedy(xx, yy[mapping], channels, 6)
            np.testing.assert_array_equal(order, expected)
            np.testing.assert_allclose(details["selected_gains"], expected_gains, rtol=3e-11, atol=2e-13)
            self.assertEqual(details["target_label_permutations"], maps)
            control_orders.append(order)
        self.assertTrue(any(not np.array_equal(control_orders[0], order) for order in control_orders[1:]))
        np.testing.assert_array_equal(x, old_x)
        np.testing.assert_array_equal(y, old_y)

    def test_inner_24_rows_exclude_validation_X_Y_and_all30_orders_are_complete(self):
        rng = np.random.default_rng(82004)
        labels, folds = design()
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 5))
        ids, channels = np.arange(32)*17+100, list(range(1, 97))
        progress = defaultdict(int)
        banks, records = selector.order_bank(x, y, labels, ids, channels, folds, "example", progress)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += rng.normal(size=(8, 288))*1e5
        changed_y[folds[0][1]] += rng.normal(size=(8, 5))*1e5
        other, other_records = selector.order_bank(changed_x, changed_y, labels, ids, channels, folds,
                                                   "example", defaultdict(int))
        self.assertEqual(len(records), 30)
        self.assertEqual(progress["selector_orderings"], 30)
        for branch in selector.BRANCHES:
            np.testing.assert_array_equal(banks[branch]["0"], other[branch]["0"])
            self.assertEqual(len(banks[branch]["outer"]), 32)
            first = next(r for r in records if r["mc_seed"] == branch and r["view"] == "0")
            second = next(r for r in other_records if r["mc_seed"] == branch and r["view"] == "0")
            self.assertEqual(first, second)

    def test_true_Y_heads_30_order_barrier_and_five_shared_full96_models(self):
        labels, folds = design()
        filename = "Mihili_CO_20140217.mat"
        pools = [list(range(12*d, 12*d+12)) for d in range(8)]
        ids = [trial for pool in pools for trial in pool[:4]]
        role = {"filename": filename, "animal": "Mihili", "fitting_pool_by_direction": pools,
                "nested_training_sets": {"32": {"training_trials": ids, "inner_folds": [
                    {"fitting_trials": [ids[i] for i in fit], "validation_trials": [ids[i] for i in valid]}
                    for fit, valid in folds]}}}
        records = {trial: {"direction_index": d} for d, pool in enumerate(pools) for trial in pool}
        source = {"common_physical_channels": list(range(1, 97))}
        rng = np.random.default_rng(82005)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 5))
        true_y = y.copy()
        progress, calls = defaultdict(int), []
        case = self
        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                self.channels = channels
                case.assertEqual(progress["selector_orderings"], 30)
            def initialize_selected(self, xx, yy, channels, guide):
                np.testing.assert_array_equal(yy, true_y)
        def model():
            progress["outer_training_refits"] += 1
            return SimpleNamespace(number=progress["outer_training_refits"])
        def full_fit(xx, yy, *args):
            np.testing.assert_array_equal(yy, true_y)
            return model()
        def selected_fit(xx, yy, dd, ff, inner, outer, count, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("scarce")
            return model(), np.asarray([3*e+b for e in outer[:count] for b in range(3)])
        def later(trajectory, total, records, pp, make_null=False):
            case.assertEqual(progress["selector_orderings"], 30)
            calls.append("future")
            return model(), object() if make_null else None
        with patch.object(experiment, "read_trials", return_value=(x, y, object())) as reads, \
                patch.object(experiment, "Trajectory", Trajectory), \
                patch.object(experiment, "fit_capacity", side_effect=full_fit), \
                patch.object(experiment, "fit_selected_rank", side_effect=selected_fit), \
                patch.object(experiment, "later_fit", side_effect=later):
            models, _, _, selections = experiment.fit_target(Path(filename), source, role, records,
                progress, lambda: None, object())
        self.assertEqual(len(selections), 30)
        self.assertEqual(len(models), 120)
        self.assertEqual(len({id(row[-1]) for row in models}), 95)
        self.assertEqual(progress["outer_training_refits"], 95)
        self.assertEqual(calls.count("scarce"), 18)
        self.assertEqual(calls.count("future"), 76)
        self.assertEqual(reads.call_count, 1)
        for n in (32, 48, 64, 80, 96):
            self.assertEqual(len({id(row[-1]) for row in models if row[1:3] == (n, 96)}), 1)
        np.testing.assert_array_equal(y, true_y)

    def test_pilot_reuse_requires_exact_recipe_support_and_complete_uncharged_work(self):
        expected = {"physical_channels": list(range(1, 97)), "initial_trial_ids": list(range(32))}
        session = {"filename": runner.PILOT, "animal": "Mihili", "source_roles": expected,
                   "all95_target_models_fixed_before_evaluation": True,
                   "all30_target_orders_fixed_before_post_initial_reads": True}
        pilot = {"status": "completed_initializer_conditional_pilot_scores_withheld", "mode": "pilot",
                 "recipe": runner.RECIPE, "policy": selector.POLICY, "control": selector.CONTROL,
                 "MC_seeds": list(selector.MC_SEEDS), "held_session_accessed": False,
                 "scientific_scores_released": False, "valid_evaluations_charged": 0,
                 "completed_session_ids": [runner.PILOT], "sessions": [session],
                 "fitting_trial_read_calls": 1248, "evaluation_trials_read": 40,
                 "fit_progress": {"outer_training_refits": 95, "inner_ridge_fits": 1900,
                                  "capacity_inner_candidate_scores": 9500, "selector_orderings": 30,
                                  "conditional_selection_steps": 960, "direction_permutation_maps": 200}}
        sources = {runner.PILOT: ({}, {"animal": "Mihili"}, {})}
        with patch.object(runner, "role_record", return_value=expected):
            self.assertIs(runner.validate_pilot(pilot, sources), session)
            for key, value in (("recipe", "other"), ("valid_evaluations_charged", 1),
                               ("scientific_scores_released", True), ("evaluation_trials_read", 39)):
                changed = copy.deepcopy(pilot)
                changed[key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    runner.validate_pilot(changed, sources)
            changed = copy.deepcopy(pilot)
            changed["sessions"][0]["source_roles"]["initial_trial_ids"][0] = 999
            with self.assertRaises(ValueError): runner.validate_pilot(changed, sources)
            changed = copy.deepcopy(pilot)
            changed["fit_progress"]["conditional_selection_steps"] -= 1
            with self.assertRaises(ValueError): runner.validate_pilot(changed, sources)

    def test_missing_initial_model_or_order_blocks_evaluation(self):
        for model_count, order_count in ((94, 30), (95, 29)):
            with self.subTest(models=model_count, orders=order_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata, roles = root/"metadata.json", root/"roles.json"
                metadata.write_text("{}")
                roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
                args = SimpleNamespace(mode="pilot", source_root=root, metadata_report=metadata, trial_roles=roles,
                    output_dir=root/"attempt", max_seconds=270, pilot_report=None,
                    target_reference=None, uniform_reference=None)
                def incomplete(path, source, role, records, progress, check, exposure):
                    progress["outer_training_refits"] += model_count
                    progress["selector_orderings"] += order_count
                    return [], {}, None, []
                with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                        patch.object(runner, "selected_metadata", return_value=({}, {}, {})), \
                        patch.object(runner, "fit_target", side_effect=incomplete), \
                        patch.object(runner, "read_trials") as evaluation:
                    with self.assertRaisesRegex(ArithmeticError, "All95"):
                        runner.main()
                evaluation.assert_not_called()

    def test_remaining_fits_all_seven_before_reading_pilot_or_historical_outcomes(self):
        class DeferredPilotObserved(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(mode="remaining", source_root=root, metadata_report=metadata, trial_roles=roles,
                output_dir=root/"attempt", max_seconds=270, pilot_report=root/"unread_pilot.json",
                target_reference=root/"unread_target.json", uniform_reference=root/"unread_uniform.json")
            fitted, evaluated, read_paths = [], [], []
            original_read = Path.read_text
            def read_text(path, *pargs, **kwargs):
                read_paths.append(path)
                if path in (args.pilot_report, args.target_reference, args.uniform_reference):
                    self.assertEqual(fitted, [name for name in runner.DEVELOPMENT if name != runner.PILOT])
                    self.assertEqual(evaluated, fitted)
                    self.assertEqual(path, args.pilot_report)
                    raise DeferredPilotObserved()
                return original_read(path, *pargs, **kwargs)
            def metadata_source(metadata, roles, filename):
                return ({"filename": filename, "common_physical_channels": list(range(1, 97))},
                        {"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                         "evaluation_trials": list(range(40))},
                        {i: {"direction_index": i//5} for i in range(40)})
            def fit(path, source, role, records, progress, check, exposure):
                self.assertNotEqual(path.name, runner.PILOT)
                self.assertFalse(any(p in read_paths for p in (args.pilot_report, args.target_reference, args.uniform_reference)))
                fitted.append(path.name)
                for key, value in (("outer_training_refits", 95), ("inner_ridge_fits", 1900),
                        ("capacity_inner_candidate_scores", 9500), ("selector_orderings", 30),
                        ("conditional_selection_steps", 960), ("direction_permutation_maps", 200)):
                    progress[key] += value
                exposure.report["fitting_trial_read_calls"] += 1248
                return [], {}, None, []
            def evaluation(path, source, role, records, ids, channels, check, exposure, guide, evaluation):
                self.assertTrue(evaluation)
                self.assertEqual(fitted[-1], path.name)
                evaluated.append(path.name)
                exposure.report["evaluation_trials_read"] += 40
                return np.zeros((40, 288)), np.ones((40, 5)), None
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "selected_metadata", side_effect=metadata_source), \
                    patch.object(runner, "fit_target", side_effect=fit), \
                    patch.object(runner, "read_trials", side_effect=evaluation), \
                    patch.object(runner, "score_target", return_value=({}, [])), \
                    patch.object(runner, "role_record", side_effect=lambda source, role: {"filename": role["filename"]}), \
                    patch.object(Path, "read_text", read_text):
                with self.assertRaises(DeferredPilotObserved):
                    runner.main()
            self.assertEqual(len(fitted), 7)
            self.assertNotIn(args.target_reference, read_paths)
            self.assertNotIn(args.uniform_reference, read_paths)


if __name__ == "__main__":
    unittest.main()
