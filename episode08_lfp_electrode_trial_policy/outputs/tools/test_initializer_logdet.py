"""Independent scientific checks for initializer-only X-residual logdet selection."""
from collections import defaultdict
import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_logdet as selector
import initializer_logdet_experiment as experiment
import development_initializer_logdet as runner


def design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(4)]
    return labels, folds


def direct_logdet_order(x, channels, maximum):
    """Rebuild the inverse from the selected set, without a recursive update."""
    chosen, gains = [], []
    for _ in range(maximum):
        columns = [3*e+b for e in chosen for b in range(3)]
        retained = x[:, columns]
        a = np.linalg.solve(np.eye(len(x))+retained@retained.T/len(x), np.eye(len(x)))
        scores = {}
        for e in range(len(channels)):
            if e in chosen:
                continue
            block = x[:, 3*e:3*e+3]
            gram = block.T@a@block/len(x)
            scores[e] = float(np.log(np.linalg.det(np.eye(3)+(gram+gram.T)/2)))
        selected = min(scores, key=lambda e: (-scores[e], channels[e]))
        chosen.append(selected)
        gains.append(scores[selected])
    return np.asarray(chosen), np.asarray(gains)


class InitializerLogdetTest(unittest.TestCase):
    def test_X_only_normalization_matches_fitting_direction_residual_population_SD(self):
        rng = np.random.default_rng(88001)
        labels, _ = design()
        x = rng.normal(size=(32, 18))+labels[:, None]*np.arange(18)[None, :]
        x[:, 0], x[:, 7] = 5., 0.
        original = x.copy()
        residual = x-np.stack([x[labels == d].mean(axis=0) for d in range(8)])[labels]
        sd = np.sqrt(np.mean((residual-residual.mean(axis=0))**2, axis=0))
        sd[sd == 0] = 1
        expected = (residual-residual.mean(axis=0))/sd
        xx, details = selector.prepare_view(x, labels, [2, 5, 8, 11, 14, 17])
        np.testing.assert_allclose(xx, expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(details["zero_SD_input_columns"], [0, 7])
        for function in (selector.prepare_view, selector.greedy_order, selector.rank_view, selector.order_bank):
            self.assertFalse({"y", "targets", "target"} & set(inspect.signature(function).parameters))

    def test_recursive_logdet_choices_match_direct_inverse_at_every_step(self):
        rng = np.random.default_rng(88002)
        x = rng.normal(size=(24, 18))
        x[:, 3:6] += .85*x[:, :3]
        channels = [1, 4, 9, 15, 22, 35]
        before = x.copy()
        order, gains, ties, negative = selector.greedy_order(x, channels, maximum=6)
        expected, expected_gains = direct_logdet_order(x, channels, 6)
        np.testing.assert_array_equal(order, expected)
        np.testing.assert_allclose(gains, expected_gains, rtol=3e-12, atol=2e-13)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(ties, [1]*6)
        self.assertEqual(negative, 0)

    def test_zero_ties_mask_chosen_groups_and_finish_exactly32_steps(self):
        order, gains, ties, negative = selector.greedy_order(np.zeros((24, 288)), list(range(1, 97)))
        np.testing.assert_array_equal(order, np.arange(32))
        np.testing.assert_array_equal(gains, np.zeros(32))
        self.assertEqual(ties, list(range(96, 64, -1)))
        self.assertEqual(len(set(order.tolist())), 32)
        self.assertEqual(negative, 0)

    def test_roundoff_negative_is_retained_but_material_negative_or_bad_sign_fails(self):
        tiny = selector.logdet_gain(-4e-16*np.eye(3))
        self.assertLess(tiny, 0)
        self.assertGreater(tiny, -128*np.finfo(float).eps)
        for gram in (-1e-6*np.eye(3), np.diag([-2., 0., 0.]), np.full((3, 3), np.nan)):
            with self.assertRaises(ArithmeticError):
                selector.logdet_gain(gram)
        with patch.object(selector, "logdet_gain", return_value=tiny):
            order, gains, ties, negatives = selector.greedy_order(np.zeros((24, 15)), [1, 2, 3, 4, 5], maximum=5)
        np.testing.assert_array_equal(order, np.arange(5))
        np.testing.assert_array_equal(gains, np.full(5, tiny))
        self.assertEqual(ties, [5, 4, 3, 2, 1])
        self.assertEqual(negatives, 15)

    def test_inner_view_cannot_use_held_X_and_all_five_orders_are_frozen(self):
        rng = np.random.default_rng(88003)
        labels, folds = design()
        x = rng.normal(size=(32, 288))
        channels = list(range(1, 97))
        progress = defaultdict(int)
        orders, records = selector.order_bank(x, labels, channels, folds, "example", progress)
        changed = x.copy()
        changed[folds[0][1]] += 1e6*rng.normal(size=(8, 288))
        other, other_records = selector.order_bank(changed, labels, channels, folds, "example", defaultdict(int))
        self.assertEqual(set(orders), set(selector.VIEWS))
        self.assertEqual(len(records), 5)
        self.assertEqual(progress["selector_orderings"], 5)
        self.assertEqual(progress["logdet_selection_steps"], 160)
        np.testing.assert_array_equal(orders["0"], other["0"])
        self.assertEqual(next(r for r in records if r["view"] == "0"),
                         next(r for r in other_records if r["view"] == "0"))
        for order in orders.values():
            self.assertEqual(len(order), 32)
            self.assertEqual(len(set(order.tolist())), 32)

    def test_five_orders_precede_future_reads_and_twenty_heads_use_true_Y(self):
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
        rng = np.random.default_rng(88004)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 5))
        true_y, progress, calls = y.copy(), defaultdict(int), []
        case = self
        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                self.channels = channels
                case.assertEqual(progress["selector_orderings"], 5)
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
            case.assertEqual(progress["selector_orderings"], 5)
            calls.append("future")
            return model(), object() if make_null else None
        with patch.object(experiment, "read_trials", return_value=(x, y, object())) as reads, \
                patch.object(experiment, "Trajectory", Trajectory), \
                patch.object(experiment, "fit_capacity", side_effect=full_fit), \
                patch.object(experiment, "fit_selected_rank", side_effect=selected_fit), \
                patch.object(experiment, "later_fit", side_effect=later):
            models, _, _, selections = experiment.fit_target(Path(filename), source, role, records,
                progress, lambda: None, object())
        self.assertEqual(len(selections), 5)
        self.assertEqual(len(models), 20)
        self.assertEqual(len({id(row[-1]) for row in models}), 20)
        self.assertEqual(progress["outer_training_refits"], 20)
        self.assertEqual(calls.count("scarce"), 3)
        self.assertEqual(calls.count("future"), 16)
        self.assertEqual(reads.call_count, 1)
        self.assertEqual(sum(row[1] == 96 for row in models), 5)
        np.testing.assert_array_equal(y, true_y)

    def test_reference_legacy_absence_and_target_denominators_full96_capacity_scores(self):
        def panel(scarce_q):
            cells = []
            for n in (32, 48, 64, 80, 96):
                for e in (8, 16, 32, 96):
                    q = .3 if e == 96 else scarce_q
                    cells.append({"training_trials": n, "electrodes": e, "null_SSE": 100.,
                        "standardized_SSE": (1-q)*100, "standardized_wrong_SSE": 80.,
                        "native_target_SSE": (1-q)*100, "Q": q, "Q_wrong": .2,
                        "pairing_increment": q-.2, "direction_only_Q": .1,
                        "delta_Q_from_direction_only": q-.1,
                        "LFP_electrode_trials": 32*96+(n-32)*e,
                        "LFP_numeric_elements": (32*96+(n-32)*e)*30,
                        "target_numeric_elements": n*10*10})
            return {"cells": cells, "fitted_models": [{**cell,
                "channels": list(range(1, cell["electrodes"]+1)), "rank": 1,
                "alpha": 1., "candidate_count": 25} for cell in cells],
                **{"mean_"+key: float(np.mean([c[key] for c in cells]))
                   for key in ("Q", "Q_wrong", "pairing_increment", "delta_Q_from_direction_only")}}
        sessions, previous = [], []
        for filename in runner.DEVELOPMENT:
            support = {"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                       "source_roles": {"filename": filename}, "target_unit_count": 10}
            marginal = panel(.4)
            for cell in marginal["cells"]:
                for key in ("null_SSE", "standardized_SSE", "standardized_wrong_SSE"):
                    cell.pop(key)
            previous.append({**support, "panels": {experiment.REPEATABILITY: panel(.3),
                experiment.UNIFORM: panel(.25), "initial_marginal": marginal},
                "seed_panels": [{"mc_seed": seed, "panel": panel(.25)} for seed in experiment.MC_SEEDS]})
            sessions.append({**support, "panels": {selector.POLICY: panel(.35)}})
        reference = {"status": "completed_two_initializer_repeatability_comparisons",
            "held_session_accessed": False, "policy": experiment.REPEATABILITY,
            "control": experiment.UNIFORM, "MC_seeds": list(experiment.MC_SEEDS),
            "recipe": "outputs/initializer_repeatability_proposal.md", "sessions": previous}
        accepted = copy.deepcopy(sessions)
        self.assertEqual(experiment.attach_references(accepted, reference), (0., 0.))
        self.assertEqual(set(accepted[0]["panels"]), set(experiment.ALL_POLICIES))
        self.assertEqual(len(accepted[0]["uniform_seed_panels"]), 5)
        self.assertNotIn("null_SSE", accepted[0]["panels"]["initial_marginal"]["cells"][0])
        for key, value in (("rank", 2), ("Q", .31), ("null_SSE", 101.), ("LFP_numeric_elements", 999)):
            changed = copy.deepcopy(sessions)
            row = next(r for r in changed[0]["panels"][selector.POLICY]["fitted_models"] if r["electrodes"] == 96)
            row[key] = value
            with self.subTest(field=key), self.assertRaises((ValueError, ArithmeticError)):
                experiment.attach_references(changed, reference)
        for policy in (experiment.REPEATABILITY, experiment.UNIFORM):
            changed = copy.deepcopy(reference)
            changed["sessions"][0]["panels"][policy]["cells"][0].pop("null_SSE")
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                experiment.attach_references(copy.deepcopy(sessions), changed)
        changed = copy.deepcopy(reference)
        next(r for r in changed["sessions"][0]["seed_panels"][0]["panel"]["fitted_models"]
             if r["electrodes"] == 96)["alpha"] = 10.
        with self.assertRaisesRegex(ArithmeticError, "capacity/support"):
            experiment.attach_references(copy.deepcopy(sessions), changed)

    def test_missing_model_or_order_blocks_evaluation(self):
        for models_count, orders_count in ((19, 5), (20, 4)):
            with self.subTest(models=models_count, orders=orders_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata, roles = root/"metadata.json", root/"roles.json"
                metadata.write_text("{}")
                roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
                args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                    reference_report=root/"closed_reference.json", output_dir=root/"attempt", max_seconds=180)
                def incomplete(path, source, role, records, progress, check, exposure):
                    progress["outer_training_refits"] += models_count
                    progress["selector_orderings"] += orders_count
                    return [], {}, None, []
                with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                        patch.object(runner, "selected_metadata", return_value=({}, {}, {})), \
                        patch.object(runner, "fit_target", side_effect=incomplete), \
                        patch.object(runner, "read_trials") as evaluation:
                    with self.assertRaisesRegex(ArithmeticError, "All20"):
                        runner.main()
                evaluation.assert_not_called()

    def test_all160_new_models_precede_the_reused_reference_read(self):
        class ReferenceReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                reference_report=root/"closed_reference.json", output_dir=root/"attempt", max_seconds=180)
            fitted, evaluated, read_paths = [], [], []
            original_read = Path.read_text
            def read_text(path, *pargs, **kwargs):
                read_paths.append(path)
                if path == args.reference_report:
                    self.assertEqual(fitted, list(runner.DEVELOPMENT))
                    self.assertEqual(evaluated, fitted)
                    raise ReferenceReached()
                return original_read(path, *pargs, **kwargs)
            def metadata_source(metadata, roles, filename):
                return ({"filename": filename, "common_physical_channels": list(range(1, 97))},
                        {"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                         "evaluation_trials": list(range(40))}, {i: {"direction_index": i//5} for i in range(40)})
            def fit(path, source, role, records, progress, check, exposure):
                self.assertNotIn(args.reference_report, read_paths)
                fitted.append(path.name)
                for key, value in (("outer_training_refits", 20), ("inner_ridge_fits", 400),
                        ("capacity_inner_candidate_scores", 2000), ("selector_orderings", 5),
                        ("logdet_selection_steps", 160), ("direction_only_models", 5)):
                    progress[key] += value
                exposure.report["fitting_trial_read_calls"] += 288
                return [], {}, None, []
            def evaluation(path, source, role, records, ids, channels, check, exposure, guide, evaluation):
                self.assertTrue(evaluation)
                self.assertEqual(fitted[-1], path.name)
                evaluated.append(path.name)
                exposure.report["evaluation_trials_read"] += 40
                return np.zeros((40, 288)), np.ones((40, 10)), None
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "selected_metadata", side_effect=metadata_source), \
                    patch.object(runner, "fit_target", side_effect=fit), \
                    patch.object(runner, "read_trials", side_effect=evaluation), \
                    patch.object(runner, "score_target", return_value={}), \
                    patch.object(runner, "role_record", side_effect=lambda source, role: {"filename": role["filename"]}), \
                    patch.object(Path, "read_text", read_text):
                with self.assertRaises(ReferenceReached):
                    runner.main()
            self.assertEqual(len(fitted), 8)


if __name__ == "__main__":
    unittest.main()
