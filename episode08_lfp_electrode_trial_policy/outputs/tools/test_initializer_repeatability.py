"""Independent checks of target-free repeatability and paid uniform selectors."""
from collections import defaultdict
import copy
import inspect
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_repeatability as selector
import initializer_repeatability_experiment as experiment
import development_initializer_repeatability as runner
from selector_training import selector_order


def design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(4)]
    return labels, folds


class InitializerRepeatabilityTest(unittest.TestCase):
    def test_raw_feature_ANOVA_matches_original_and_constant_bands_keep_their_weight(self):
        labels, _ = design()
        direction_signal = labels.astype(float)
        noise = np.tile([-1., 1., -1., 1.], 8)
        x = np.column_stack((direction_signal, np.ones(32), noise,
                             direction_signal, 3*direction_signal, -direction_signal,
                             np.ones(32), np.zeros(32), np.full(32, 8.)))
        scores, _ = selector.repeatability_scores(x, labels, [2, 7, 19])
        np.testing.assert_allclose(scores, [1/3, 1., 0.], atol=1e-14)
        rng = np.random.default_rng(81001)
        x = rng.normal(size=(32, 288))+labels[:, None]*rng.normal(size=(1, 288))
        original = x.copy()
        scores, _ = selector.repeatability_scores(x, labels, list(range(1, 97)))
        sd = x.std(axis=0)
        z = (x-x.mean(axis=0))/np.where(sd == 0, 1., sd)
        means = np.stack([z[labels == d].mean(axis=0) for d in range(8)])
        between = 4*np.sum((means-z.mean(axis=0))**2, axis=0)/7
        within = sum(np.sum((z[labels == d]-means[d])**2, axis=0) for d in range(8))/24
        tau = np.maximum((between-within)/4, 0)
        expected = np.divide(tau, tau+within, out=np.zeros_like(tau), where=tau+within > 0)
        np.testing.assert_allclose(scores, expected.reshape(96, 3).mean(axis=1), rtol=1e-12, atol=1e-14)
        order, _ = selector.rank_view(x, labels, list(range(1, 97)), "example", "outer")
        previous = selector_order(x, rng.normal(size=(32, 4)), labels, "direction_repeatability", len(order))
        np.testing.assert_array_equal(order, previous)
        np.testing.assert_array_equal(x, original)

    def test_selector_has_no_target_input_and_ties_keep_ascending_physical_IDs(self):
        labels, _ = design()
        self.assertEqual(list(inspect.signature(selector.repeatability_scores).parameters),
                         ["x", "directions", "channels"])
        self.assertNotIn("y", inspect.signature(selector.rank_view).parameters)
        self.assertNotIn("y", inspect.signature(selector.order_bank).parameters)
        channels = [2, 8, 14, 30]
        scores, _ = selector.repeatability_scores(np.zeros((32, 12)), labels, channels)
        order, _ = selector.rank_view(np.zeros((32, 12)), labels, channels, "example", "outer")
        np.testing.assert_array_equal(scores, np.zeros(4))
        np.testing.assert_array_equal(order, np.arange(4))

    def test_uniform_order_is_one_canonical_physical_shuffle_independent_of_ties(self):
        channels = list(range(3, 291, 3))
        order, record = selector.uniform_order(channels, "example", "outer", 2)
        expected = channels.copy()
        random.Random("ep08-initial-uniform-order-v1|20261004|example|outer|2").shuffle(expected)
        self.assertEqual([channels[int(i)] for i in order], expected)
        self.assertEqual(sorted(order.tolist()), list(range(96)))
        labels, _ = design()
        tied, _ = selector.rank_view(np.zeros((32, 288)), labels, channels, "example", "outer", 2)
        variable, _ = selector.rank_view(np.random.default_rng(81002).normal(size=(32, 288)),
                                       labels, channels, "example", "outer", 2)
        np.testing.assert_array_equal(tied, variable)
        np.testing.assert_array_equal(tied, order[:len(tied)])
        for count in (8, 16, 32):
            self.assertEqual(len(set(order[:count])), count)
            self.assertTrue(set(order[:count]) <= set(order[:32]))
        calls = []
        class Identity:
            def seed(self, literal, version): calls.append((literal, version))
            def shuffle(self, values): calls.append(tuple(values))
        with patch.object(selector.random, "Random", Identity):
            identity, _ = selector.uniform_order(channels, "example", "2", 4)
        np.testing.assert_array_equal(identity, np.arange(96))
        self.assertEqual(calls, [("ep08-initial-uniform-order-v1|20261004|example|2|4", 2), tuple(channels)])

    def test_inner_view_excludes_validation_rows_and_freezes_30_orders(self):
        labels, folds = design()
        x = np.random.default_rng(81003).normal(size=(32, 288))
        channels = list(range(1, 97))
        progress = defaultdict(int)
        banks, records = selector.order_bank(x, labels, channels, folds, "example", progress)
        altered = x.copy()
        altered[folds[0][1]] += np.random.default_rng(81004).normal(size=(8, 288))*1e6
        other, other_records = selector.order_bank(altered, labels, channels, folds, "example", defaultdict(int))
        self.assertEqual(len(records), 30)
        self.assertEqual(progress["selector_orderings"], 30)
        for branch in selector.BRANCHES:
            np.testing.assert_array_equal(banks[branch]["0"], other[branch]["0"])
            self.assertEqual(len(banks[branch]["outer"]), 32)
        # Uniform branches cannot depend on any feature values, even outer rows.
        for branch in selector.MC_SEEDS:
            for view in selector.VIEWS:
                np.testing.assert_array_equal(banks[branch][view], other[branch][view])

    def test_orders_precede_future_reads_true_targets_fit_and_all96_models_are_shared(self):
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
        rng = np.random.default_rng(81005)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 10))
        true_y = y.copy()
        progress, calls = defaultdict(int), []
        case = self
        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                self.channels = channels
                case.assertEqual(progress["selector_orderings"], 30)
            def initialize_selected(self, xx, yy, channels, guide):
                np.testing.assert_array_equal(yy, true_y)
                case.assertEqual(xx.shape, (32, 288))
        def model():
            progress["outer_training_refits"] += 1
            return SimpleNamespace(number=progress["outer_training_refits"])
        def full_fit(xx, yy, *args):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("full")
            return model()
        def selected_fit(xx, yy, dd, ff, inner, outer, count, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("scarce")
            columns = np.array([3*e+b for e in outer[:count] for b in range(3)])
            return model(), columns
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
        self.assertEqual(progress["outer_training_refits"], 95)
        self.assertEqual(len(models), 120)
        self.assertEqual(len({id(item[-1]) for item in models}), 95)
        self.assertEqual(calls.count("scarce"), 18)
        self.assertEqual(calls.count("future"), 76)
        self.assertEqual(reads.call_count, 1)
        for total in (32, 48, 64, 80, 96):
            full = [item for item in models if item[1:3] == (total, 96)]
            self.assertEqual(len(full), 6)
            self.assertEqual(len({id(item[-1]) for item in full}), 1)
        np.testing.assert_array_equal(y, true_y)

    def test_runner_rejects_missing_model_or_order_before_evaluation(self):
        for models_count, orders_count in ((94, 30), (95, 29)):
            with self.subTest(models=models_count, orders=orders_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata, roles = root/"metadata.json", root/"roles.json"
                metadata.write_text("{}")
                roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
                args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                    reference_report=root/"closed_initializer.json", acquired_reference=root/"closed_acquired.json",
                    output_dir=root/"failed_attempt", max_seconds=270)
                def incomplete(path, source, role, records, progress, check, exposure):
                    progress["outer_training_refits"] += models_count
                    progress["selector_orderings"] += orders_count
                    return [], {}, None, []
                with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                        patch.object(runner, "selected_metadata", return_value=({}, {}, {})), \
                        patch.object(runner, "fit_target", side_effect=incomplete), \
                        patch.object(runner, "read_trials") as evaluation:
                    with self.assertRaisesRegex(ArithmeticError, "All95"):
                        runner.main()
                evaluation.assert_not_called()

    def test_score_path_averages_uniform_seed_SSE_and_keeps_all20_cells(self):
        channels = list(range(1, 97))
        labels = np.repeat(np.arange(8), 5)
        x, y = np.zeros((40, 288)), np.ones((40, 2))
        statistics = SimpleNamespace(target_mean=np.zeros(2), target_sd=np.ones(2))
        null = SimpleNamespace(direction_target_mean=np.zeros((8, 2)), target_mean=np.zeros(2),
                               target_sd=np.ones(2))
        models = []
        for branch in selector.BRANCHES:
            for total in (32, 48, 64, 80, 96):
                for count in (8, 16, 32, 96):
                    amplitude = 0 if branch is None or count == 96 else branch+1
                    fitted = SimpleNamespace(statistics=statistics, rank=1, alpha=1.,
                                             candidate_count=25, amplitude=amplitude)
                    models.append((branch, total, count, tuple(channels[:count]), fitted))
        with patch.object(experiment, "predict_capacity", side_effect=lambda model, xx, dd: y+model.amplitude):
            panels, seeds = experiment.score_target(models, {n: null for n in (32, 48, 64, 80, 96)},
                                                     x, y, labels, channels)
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(panels[selector.CONTROL]["cells"]), 20)
        self.assertAlmostEqual(panels[selector.POLICY]["mean_Q"], 1.)
        self.assertAlmostEqual(panels[selector.CONTROL]["mean_Q"], -7.25)
        for cell in panels[selector.CONTROL]["cells"]:
            expected = 0. if cell["electrodes"] == 96 else 880.
            self.assertAlmostEqual(cell["standardized_SSE"], expected)
            self.assertEqual(cell["LFP_numeric_elements"], cell["LFP_electrode_trials"]*30)

    def test_reference_full96_replay_rejects_changed_capacity_or_score(self):
        def panel():
            cells = [{"training_trials": n, "electrodes": e, "null_SSE": 100.,
                      "standardized_SSE": 70., "standardized_wrong_SSE": 80.,
                      "native_target_SSE": 70., "Q": .3, "Q_wrong": .2,
                      "pairing_increment": .1, "direction_only_Q": .1,
                      "delta_Q_from_direction_only": .2,
                      "LFP_electrode_trials": 32*96+(n-32)*e,
                      "LFP_numeric_elements": (32*96+(n-32)*e)*30,
                      "target_numeric_elements": n*10*10}
                     for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)]
            return {"cells": cells, "fitted_models": [{**cell,
                    "channels": list(range(1, cell["electrodes"]+1)),
                    "rank": 1, "alpha": 1., "candidate_count": 25} for cell in cells]}
        sessions, historical, acquired = [], [], []
        for filename in runner.DEVELOPMENT:
            animal = "Mihili" if filename.startswith("Mihili") else "Chewie"
            old_panels = {name: panel() for name in ("initial_marginal", experiment.PREVIOUS_CONTROL, "fixed_random")}
            support = {"filename": filename, "animal": animal,
                       "source_roles": {"filename": filename}, "target_unit_count": 10}
            historical.append({**support, "panels": old_panels,
                               "seed_panels": [{"mc_seed": seed, "panel": panel()} for seed in selector.MC_SEEDS]})
            acquired.append({**support, "panels": copy.deepcopy(old_panels)})
            sessions.append({**support, "panels": {selector.POLICY: panel(), selector.CONTROL: panel()},
                             "seed_panels": [{"mc_seed": seed, "panel": panel()} for seed in selector.MC_SEEDS],
                             "selection_records": []})
        previous = {"status": "completed_one_initializer_target_label_control", "held_session_accessed": False,
                    "control": experiment.PREVIOUS_CONTROL, "MC_seeds": list(selector.MC_SEEDS),
                    "sessions": historical}
        original = {"status": "completed_one_acquired_only_comparison_with_random_reference",
                    "recipe": "outputs/acquired_only_proposal.md", "held_session_accessed": False,
                    "sessions": acquired}
        self.assertEqual(experiment.attach_references(copy.deepcopy(sessions), previous, original), (0., 0.))
        for key, value in (("rank", 2), ("Q", .31)):
            changed = copy.deepcopy(sessions)
            row = next(row for row in changed[0]["seed_panels"][0]["panel"]["fitted_models"]
                       if row["electrodes"] == 96)
            row[key] = value
            with self.subTest(field=key), self.assertRaises(ArithmeticError):
                experiment.attach_references(changed, previous, original)


if __name__ == "__main__":
    unittest.main()
