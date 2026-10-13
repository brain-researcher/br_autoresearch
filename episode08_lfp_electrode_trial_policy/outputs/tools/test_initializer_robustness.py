"""Independent immutable-role, selector and across-initializer checks."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_robustness_roles as roles_module
import initializer_robustness as selector
import initializer_robustness_experiment as experiment
import development_initializer_robustness as runner
from acquired_only_reader import CHECKPOINTS
from acquired_rank_training import conditioned_folds
from development_reader import selected_metadata


def original_packets():
    sources, roles = [], []
    for name in roles_module.DEVELOPMENT:
        pools = [list(range(100*d+11, 100*d-1, -1))+[100*d+20, 100*d+21]
                 for d in range(8)]
        evaluation = [100*d+70+j for d in range(8) for j in range(5)]
        records = [{"trial_index_zero_based": trial, "direction_index": d}
                   for d, pool in enumerate(pools) for trial in [*pool, *range(100*d+70, 100*d+75)]]
        animal = "Mihili" if name.startswith("Mihili") else "Chewie"
        source = {"filename": name, "session_role": "development", "animal": animal,
                  "common_physical_channels": list(range(1, 97)), "metadata_eligible_trials": records}
        nested = {}
        for n in CHECKPOINTS:
            training = [trial for pool in pools for trial in pool[:n//8]]
            folds = []
            for k in range(4):
                valid = [trial for pool in pools for position, trial in enumerate(pool[:n//8])
                         if position % 4 == k]
                folds.append({"fold_index": k, "fitting_trials": [t for t in training if t not in valid],
                              "validation_trials": valid})
            nested[str(n)] = {"training_trials": training, "inner_folds": folds}
        sources.append(source)
        roles.append({"filename": name, "session_role": "development", "animal": animal,
                      "fitting_pool_by_direction": pools, "evaluation_trials": evaluation,
                      "nested_training_sets": nested})
    return {"sessions": sources}, {"sessions": roles}


def initializer_arrays(seed=141001):
    metadata, original = original_packets()
    source, old, records = selected_metadata(metadata, original, roles_module.DEVELOPMENT[0])
    role, _ = roles_module.derive_role(source, old, records, 1)
    ids = role["nested_training_sets"]["32"]["training_trials"]
    positions = {trial: i for i, trial in enumerate(ids)}
    folds = [(np.asarray([positions[t] for t in f["fitting_trials"]]),
              np.asarray([positions[t] for t in f["validation_trials"]]))
             for f in role["nested_training_sets"]["32"]["inner_folds"]]
    labels = np.asarray([records[t]["direction_index"] for t in ids])
    rng = np.random.default_rng(seed)
    return source, role, records, np.asarray(ids), labels, folds, rng.normal(size=(32, 288)), rng.normal(size=(32, 3))


def scored_orders():
    labels, channels = np.repeat(np.arange(8), 5), list(range(1, 97))
    stats = SimpleNamespace(target_mean=np.zeros(2), target_sd=np.ones(2))
    null = SimpleNamespace(direction_target_mean=np.zeros((8, 2)),
                           target_mean=np.zeros(2), target_sd=np.ones(2))
    results = []
    for order, target in enumerate((1., 2., 4.)):
        models = []
        for seed in selector.BRANCHES:
            for n in CHECKPOINTS:
                for e in (8, 16, 32, 96):
                    value = 1. if seed is None or e == 96 else float(seed+1)
                    fitted = SimpleNamespace(statistics=stats, rank=1+order,
                        alpha=1.+order, candidate_count=25, value=value)
                    models.append((seed, n, e, tuple(channels[:e]), fitted))
        with patch.object(experiment, "predict_capacity", side_effect=lambda model, x, labels:
                          np.full((len(x), 2), model.value)):
            panels, seeds = experiment.score_target(models, {n: null for n in CHECKPOINTS},
                np.zeros((40, 288)), np.full((40, 2), target), labels, channels)
        results.append({"initializer_order": order, "panels": panels, "seed_panels": seeds})
    return results


def runner_arguments(root):
    metadata, original = original_packets()
    prepared = roles_module.prepare_roles(metadata, original)
    args = SimpleNamespace(source_root=root, metadata_report=root/"metadata.json",
        trial_roles=root/"roles.json", prepared_roles=root/"prepared.json",
        reference_report=root/"closed_reference.json", output_dir=root/"attempt", max_seconds=600)
    for path, packet in ((args.metadata_report, metadata), (args.trial_roles, original),
                         (args.prepared_roles, prepared)):
        path.write_text(json.dumps(packet))
    return args


def add_work(progress, exposure, models=95, orders=30):
    for key, value in (("outer_training_refits", models), ("inner_ridge_fits", 1900),
            ("capacity_inner_candidate_scores", 9500), ("selector_orderings", orders),
            ("marginal_score_solves", 2880), ("residual_normalization_views", 5),
            ("direction_permutation_maps", 200), ("direction_only_models", 5)):
        progress[key] += value
    exposure.report["fitting_trial_read_calls"] += 1248


class InitializerRobustnessTest(unittest.TestCase):
    def test_three_ordinary_canonical_orders_preserve_exact_whitelist_and_evaluation(self):
        metadata, original = original_packets()
        before = copy.deepcopy(original)
        prepared = roles_module.prepare_roles(metadata, original)
        self.assertEqual(prepared["initializer_orders"], [0, 1, 2])
        self.assertEqual(len(prepared["orders"]), 3)
        self.assertIs(roles_module.validate_prepared(prepared, metadata, original), prepared)
        for order in prepared["orders"]:
            self.assertEqual(len(order["sessions"]), 8)
            for role, old in zip(order["sessions"], original["sessions"]):
                self.assertEqual(role["evaluation_trials"], old["evaluation_trials"])
                self.assertEqual(set(role["nested_training_sets"]["96"]["training_trials"]),
                                 set(old["nested_training_sets"]["96"]["training_trials"]))
                for d, queue in enumerate(role["fitting_pool_by_direction"]):
                    expected = sorted(old["fitting_pool_by_direction"][d][:12])
                    rng = random.Random()
                    rng.seed(f"ep08-initializer-robustness-order-v1|20261004|{role['filename']}|{d}|{order['initializer_order']}", version=2)
                    rng.shuffle(expected)
                    self.assertEqual(queue, expected)
                    self.assertNotIn(100*d+20, queue)
        self.assertEqual(original, before)
        literals = []
        class Identity:
            def seed(self, literal, version): literals.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(roles_module.random, "Random", Identity):
            identity = roles_module.prepare_roles(metadata, original)
        self.assertEqual(len(literals), 3*8*8)
        self.assertTrue(all(version == 2 for _, version in literals))
        for packet in identity["orders"]:
            for role in packet["sessions"]:
                self.assertEqual(role["fitting_pool_by_direction"], [list(range(100*d, 100*d+12)) for d in range(8)])
        self.assertEqual(identity["orders"][0]["sessions"][0]["fitting_pool_by_direction"],
                         identity["orders"][2]["sessions"][0]["fitting_pool_by_direction"])

    def test_role_corruption_fails_closed_and_main_folds_have_no_direction_offset(self):
        metadata, original = original_packets()
        prepared = roles_module.prepare_roles(metadata, original)
        for kind in ("whitelist", "evaluation", "queue", "fold", "map",
                     "prefix48", "prefix80", "fold48", "fold80"):
            changed = copy.deepcopy(prepared)
            role = changed["orders"][0]["sessions"][0]
            if kind == "whitelist": role["nested_training_sets"]["96"]["training_trials"][0] = 99999
            elif kind == "evaluation": role["evaluation_trials"][0] = 99999
            elif kind == "queue": role["fitting_pool_by_direction"][0].reverse()
            elif kind == "fold": role["nested_training_sets"]["32"]["inner_folds"][0]["validation_trials"].reverse()
            elif kind.startswith("prefix"): role["nested_training_sets"][kind[6:]]["training_trials"].reverse()
            elif kind.startswith("fold"): role["nested_training_sets"][kind[4:]]["inner_folds"][0]["validation_trials"].reverse()
            else: changed["orders"][0]["queue_records"][0]["queues"][0]["seed_literal"] += "wrong"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                roles_module.validate_prepared(changed, metadata, original)
        for packet in prepared["orders"]:
            for role in packet["sessions"]:
                pools = role["fitting_pool_by_direction"]
                initializer = role["nested_training_sets"]["32"]["training_trials"]
                for k, fold in enumerate(role["nested_training_sets"]["32"]["inner_folds"]):
                    self.assertEqual(fold["validation_trials"], [pool[k] for pool in pools])
                    self.assertEqual(len(fold["fitting_trials"]), 24)
                    self.assertEqual(set(fold["fitting_trials"]) | set(fold["validation_trials"]), set(initializer))
                # Later decoder CV deliberately differs from metadata folds.
                labels = np.repeat(np.arange(8), 6)
                initial_mask = np.tile(np.arange(6) < 4, 8)
                post = np.tile(np.arange(6)-4, 8)
                initial, folds = conditioned_folds(labels, initial_mask, post)
                for k, (fit, valid) in enumerate(folds):
                    self.assertTrue(set(initial).issubset(fit))
                    np.testing.assert_array_equal(valid, np.flatnonzero(~initial_mask & ((post+labels) % 4 == k)))

    def test_original_pool_tail_and_evaluation_cannot_become_new_initializer_support(self):
        metadata, original = original_packets()
        source, role, records = selected_metadata(metadata, original, roles_module.DEVELOPMENT[0])
        for replacement in (role["fitting_pool_by_direction"][0][12], role["evaluation_trials"][0]):
            bad = copy.deepcopy(role)
            bad["fitting_pool_by_direction"][0][0] = replacement
            with self.assertRaises(ValueError):
                roles_module.derive_role(source, bad, records, 0)

    def test_residual_marginal_oracle_new_canonical_target_maps_and_identity_draws(self):
        source, role, records, ids, labels, folds, x, y = initializer_arrays()
        channels = [2, 7, 11]
        x = x[:, :9]+labels[:, None]*np.arange(1, 10)
        y = y+labels[:, None]*np.array([3., 17., 50.])
        x[:, 2], y[:, 1] = 7., 9.
        prepared = selector.prepare_view(x, y, labels, channels)
        mx = np.stack([x[labels == d].mean(axis=0) for d in range(8)])
        my = np.stack([y[labels == d].mean(axis=0) for d in range(8)])
        xr = x-mx[labels]
        xs, ys = xr.std(axis=0), y.std(axis=0)
        xs[xs == 0], ys[ys == 0] = 1., 1.
        z = (xr-xr.mean(axis=0))/xs
        target = (y-my[labels])/ys
        target -= target.mean(axis=0)
        np.testing.assert_array_equal(prepared[0].reshape(x.shape), z)
        np.testing.assert_array_equal(prepared[1], target)
        name = role["filename"]
        permutation, maps = selector.within_direction_map(ids, labels, name, "outer", 3, 1)
        np.testing.assert_array_equal(labels[permutation], labels)
        self.assertEqual(sorted(permutation.tolist()), list(range(32)))
        for row in maps:
            expected = row["recipient_trial_ids"].copy()
            literal = f"ep08-initializer-robustness-target-label-v1|20261004|{name}|1|outer|3|{row['direction']}"
            rng = random.Random()
            rng.seed(literal, version=2)
            rng.shuffle(expected)
            self.assertEqual(row["seed_literal"], literal)
            self.assertEqual(row["donor_trial_ids"], expected)
        reversed_ids = np.arange(32)[::-1]
        other_map, other_records = selector.within_direction_map(ids[reversed_ids], labels[reversed_ids], name, "outer", 3, 1)
        self.assertEqual(maps, other_records)
        np.testing.assert_array_equal(ids[reversed_ids][other_map], ids[permutation][reversed_ids])
        order, gains, got = selector.rank_prepared(prepared, labels, ids, channels, name, "outer", 3, 1)
        expected_scores = []
        for e in range(3):
            block = z[:, 3*e:3*e+3]
            g, c = block.T@block/32, block.T@target[permutation]/32
            expected_scores.append(np.sum(c*np.linalg.solve(np.eye(3)+g, c))/3)
        np.testing.assert_allclose(gains, expected_scores, rtol=2e-13, atol=2e-14)
        self.assertEqual(got, maps)
        np.testing.assert_array_equal(order, np.lexsort((channels, -np.asarray(expected_scores))))
        class Identity:
            def seed(self, literal, version): self.version = version
            def shuffle(self, values): self.asserted = self.version == 2
        with patch.object(selector.random, "Random", Identity):
            identity_order, identity_scores, _ = selector.rank_prepared(prepared, labels, ids, channels, name, "outer", 3, 1)
        true_order, true_scores, _ = selector.rank_prepared(prepared, labels, ids, channels, name, "outer", None, 1)
        np.testing.assert_array_equal(identity_order, true_order)
        np.testing.assert_array_equal(identity_scores, true_scores)
        tied, zero, _ = selector.rank_view(np.zeros_like(x), np.zeros_like(y), labels, ids,
                                          channels, name, "outer", None, 1)
        np.testing.assert_array_equal(tied, np.arange(3))
        np.testing.assert_array_equal(zero, np.zeros(3))

    def test_all30_orders_fit_only_their_original_main_view_and_keep_fixed_work(self):
        source, role, records, ids, labels, folds, x, y = initializer_arrays(141002)
        progress = defaultdict(int)
        banks, rows = selector.order_bank(x, y, labels, ids, source["common_physical_channels"],
            folds, role["filename"], 1, progress)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += 1e6
        changed_y[folds[0][1]] -= 1e6
        other, other_rows = selector.order_bank(changed_x, changed_y, labels, ids, source["common_physical_channels"],
            folds, role["filename"], 1, defaultdict(int))
        for key, value in (("selector_orderings", 30), ("direction_permutation_maps", 200),
                           ("marginal_score_solves", 2880), ("residual_normalization_views", 5)):
            self.assertEqual(progress[key], value)
        for branch in selector.BRANCHES:
            np.testing.assert_array_equal(banks[branch]["0"], other[branch]["0"])
            first = next(r for r in rows if r["mc_seed"] == branch and r["view"] == "0")
            second = next(r for r in other_rows if r["mc_seed"] == branch and r["view"] == "0")
            self.assertEqual(first, second)
            self.assertEqual(set(first["fitting_trial_ids"]), set(ids[folds[0][0]]))
        self.assertEqual(len(rows), 30)

    def test_each_order_reads_fresh_initializer_and_only_true_Y_enters_all95_heads(self):
        source, role, records, ids, labels, folds, x, y = initializer_arrays(141003)
        progress, calls = defaultdict(int), []
        case = self
        active = {"order": None, "y": None}
        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                case.assertEqual(progress["selector_orderings"], 30*(active["order"]+1))
            def initialize_selected(self, xx, yy, channels, guide):
                np.testing.assert_array_equal(yy, active["y"])
        def model():
            progress["outer_training_refits"] += 1
            return SimpleNamespace(number=progress["outer_training_refits"])
        def initial_read(path, source, role, records, selected_ids, channels, check, exposure):
            calls.append(("initial", active["order"]))
            return x.copy(), active["y"].copy(), (np.arange(96), np.arange(3))
        def full_fit(xx, yy, *args):
            np.testing.assert_array_equal(yy, active["y"])
            return model()
        def selected_fit(xx, yy, dd, ff, inner, outer, count, pp):
            np.testing.assert_array_equal(yy, active["y"])
            calls.append(("scarce", active["order"]))
            return model(), np.asarray([3*e+b for e in outer[:count] for b in range(3)])
        def later(trajectory, total, records, pp, make_null=False):
            case.assertEqual(progress["selector_orderings"], 30*(active["order"]+1))
            calls.append(("future", active["order"]))
            return model(), object() if make_null else None
        previous_model_ids = set()
        metadata, original = original_packets()
        _, original_role, _ = selected_metadata(metadata, original, role["filename"])
        with patch.object(experiment, "read_trials", side_effect=initial_read), \
                patch.object(experiment, "Trajectory", Trajectory), \
                patch.object(experiment, "fit_capacity", side_effect=full_fit), \
                patch.object(experiment, "fit_selected_rank", side_effect=selected_fit), \
                patch.object(experiment, "later_fit", side_effect=later):
            retained = []
            for order in roles_module.ORDERS:
                active.update(order=order, y=y+1000*order)
                current_role, _ = roles_module.derive_role(source, original_role, records, order)
                models, _, _, selections = experiment.fit_target(Path(role["filename"]), source, current_role,
                    records, progress, lambda: None, object(), order)
                retained.append(models)  # Prevent object-ID recycling between conditions.
                actual_ids = {id(row[-1]) for row in models}
                self.assertEqual(len(models), 120)
                self.assertEqual(len(actual_ids), 95)
                self.assertFalse(actual_ids & previous_model_ids)
                previous_model_ids.update(actual_ids)
                self.assertEqual(len(selections), 30)
        self.assertEqual(progress["outer_training_refits"], 285)
        self.assertEqual(progress["selector_orderings"], 90)
        for order in roles_module.ORDERS:
            self.assertEqual(calls.count(("initial", order)), 1)
            self.assertEqual(calls.count(("scarce", order)), 18)
            self.assertEqual(calls.count(("future", order)), 76)

    def test_control_mean_SSE_precedes_equal_mean_Q_across_unequal_order_denominators(self):
        results = scored_orders()
        for result, denominator, loss in zip(results, (80., 320., 1280.), (480., 240., 240.)):
            self.assertEqual(experiment.validate_order_panels(result["panels"], result["seed_panels"], 2), 0.)
            cell = next(c for c in result["panels"][selector.CONTROL]["cells"] if c["training_trials"] == 32 and c["electrodes"] == 8)
            self.assertEqual(cell["null_SSE"], denominator)
            self.assertEqual(cell["standardized_SSE"], loss)
            self.assertEqual(cell["Q"], 1-loss/denominator)
        panels, seeds = experiment.average_orders(results)
        control = next(c for c in panels[selector.CONTROL]["cells"] if c["training_trials"] == 32 and c["electrodes"] == 8)
        self.assertEqual(control["Q"], np.mean([-5., .25, .8125]))
        self.assertNotAlmostEqual(control["Q"], 1-(480+240+240)/(80+320+1280))
        self.assertEqual(len(seeds), 5)
        for panel in [*panels.values(), *(s["panel"] for s in seeds)]:
            self.assertNotIn("fitted_models", panel)
            for cell in panel["cells"]:
                self.assertFalse({"null_SSE", "standardized_SSE", "standardized_wrong_SSE",
                                  "native_target_SSE", "rank", "alpha", "channels", "candidate_count"} & cell.keys())
        changed = copy.deepcopy(results)
        changed[1]["panels"][selector.POLICY]["cells"][0]["Q"] = float("nan")
        with self.assertRaises(ArithmeticError): experiment.average_orders(changed)
        with self.assertRaises(ValueError): experiment.average_orders(results[:2])

    def test_full96_identity_is_exact_within_order_and_not_imposed_across_orders(self):
        results = scored_orders()
        self.assertNotEqual(results[0]["panels"][selector.POLICY]["cells"][-1]["Q"],
                            results[2]["panels"][selector.POLICY]["cells"][-1]["Q"])
        for result in results:
            self.assertEqual(experiment.validate_order_panels(result["panels"], result["seed_panels"], 2), 0.)
        experiment.average_orders(results)
        for field, value in (("Q", 1.-5e-12), ("rank", 2), ("alpha", 100.), ("null_SSE", 81.)):
            changed = copy.deepcopy(results[0])
            row = next(row for row in changed["seed_panels"][4]["panel"]["fitted_models"] if row["electrodes"] == 96)
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ArithmeticError):
                experiment.validate_order_panels(changed["panels"], changed["seed_panels"], 2)

    def test_original_context_preserves_legacy_absence_without_false_new_score_replay(self):
        metadata, roles = original_packets()
        sources = {name: selected_metadata(metadata, roles, name) for name in roles_module.DEVELOPMENT}
        results = scored_orders()
        panels, seeds = experiment.average_orders(results)
        sessions, previous_sessions = [], []
        for name in roles_module.DEVELOPMENT:
            source, role, _ = sources[name]
            sessions.append({"filename": name, "animal": role["animal"], "target_unit_count": 2,
                             "panels": copy.deepcopy(panels), "seed_panels": copy.deepcopy(seeds)})
            old_true = copy.deepcopy(results[2]["panels"][selector.POLICY])
            for cell in [*old_true["cells"], *old_true["fitted_models"]]:
                for key in ("null_SSE", "standardized_SSE", "standardized_wrong_SSE"):
                    cell.pop(key)
            previous_sessions.append({"filename": name, "animal": role["animal"], "target_unit_count": 2,
                "source_roles": experiment.role_record(source, role), "panels": {
                    "initial_marginal": old_true,
                    experiment.PREVIOUS_CONTROL: copy.deepcopy(results[2]["panels"][selector.CONTROL])}})
        previous = {"status": "completed_one_initializer_target_label_control",
            "recipe": "outputs/initializer_target_label_control_proposal.md", "held_session_accessed": False,
            "control": experiment.PREVIOUS_CONTROL, "MC_seeds": list(selector.MC_SEEDS), "sessions": previous_sessions}
        experiment.attach_original_context(sessions, previous, sources)
        legacy = sessions[0]["original_initializer_context"]["initial_marginal"]["cells"][0]
        self.assertNotIn("null_SSE", legacy)
        self.assertNotEqual(legacy["Q"], sessions[0]["panels"][selector.POLICY]["cells"][0]["Q"])
        for kind in ("roles", "recipe", "missing_cell", "unit_count"):
            changed = copy.deepcopy(previous)
            if kind == "roles": changed["sessions"][0]["source_roles"]["evaluation_trial_ids"][0] = -1
            elif kind == "recipe": changed["recipe"] = "unadopted"
            elif kind == "missing_cell": changed["sessions"][0]["panels"]["initial_marginal"]["cells"].pop()
            else: changed["sessions"][0]["target_unit_count"] = 3
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                experiment.attach_original_context(copy.deepcopy(sessions), changed, sources)

    def test_incomplete_third_order_or_mismatched_guide_blocks_any_evaluation_read(self):
        for kind in ("model", "order", "guide_value", "guide_length"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args = runner_arguments(Path(directory))
                fitted = []
                def fit(path, source, role, records, progress, check, exposure, order):
                    fitted.append(order)
                    add_work(progress, exposure, 94 if order == 2 and kind == "model" else 95,
                             29 if order == 2 and kind == "order" else 30)
                    guide = (np.arange(96), np.arange(2))
                    if order == 2 and kind == "guide_value": guide = (np.arange(96)+1, np.arange(2))
                    if order == 2 and kind == "guide_length": guide = (np.arange(96),)
                    return [], {}, guide, []
                with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                        patch.object(runner, "fit_target", side_effect=fit), \
                        patch.object(runner, "read_trials") as evaluation:
                    with self.assertRaises(ArithmeticError): runner.main()
                evaluation.assert_not_called()
                self.assertEqual(fitted, [0, 1, 2])

    def test_all2280_fits_precede_original_context_and_each285_precedes_single_eval(self):
        class ReferenceReached(Exception): pass
        results = scored_orders()
        with tempfile.TemporaryDirectory() as directory:
            args = runner_arguments(Path(directory))
            fitted, evaluated, read_paths = [], [], []
            original_read = Path.read_text
            def read_text(path, *pargs, **kwargs):
                read_paths.append(path)
                if path == args.reference_report:
                    self.assertEqual(fitted, [(name, order) for name in roles_module.DEVELOPMENT for order in (0, 1, 2)])
                    self.assertEqual(evaluated, list(roles_module.DEVELOPMENT))
                    raise ReferenceReached()
                return original_read(path, *pargs, **kwargs)
            def fit(path, source, role, records, progress, check, exposure, order):
                self.assertNotIn(args.reference_report, read_paths)
                self.assertEqual(role["initializer_order"], order)
                fitted.append((path.name, order))
                add_work(progress, exposure)
                return order, {}, (np.arange(96), np.arange(2)), []
            def evaluation(path, source, role, records, ids, channels, check, exposure, guide, evaluation):
                self.assertTrue(evaluation)
                self.assertEqual([o for name, o in fitted if name == path.name], [0, 1, 2])
                self.assertNotIn(path.name, evaluated)
                evaluated.append(path.name)
                exposure.report["evaluation_trials_read"] += 40
                return np.zeros((40, 288)), np.ones((40, 2)), guide
            def score(models, *args):
                current = copy.deepcopy(results[models])
                return current["panels"], current["seed_panels"]
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "fit_target", side_effect=fit), \
                    patch.object(runner, "read_trials", side_effect=evaluation), \
                    patch.object(runner, "score_target", side_effect=score), \
                    patch.object(Path, "read_text", read_text):
                with self.assertRaises(ReferenceReached): runner.main()
            self.assertEqual(len(fitted), 24)
            self.assertEqual(len(evaluated), 8)


if __name__ == "__main__":
    unittest.main()
