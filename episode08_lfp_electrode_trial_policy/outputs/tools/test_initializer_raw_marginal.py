"""Independent raw-coordinate, retained-direction and acquisition checks."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_raw_marginal as selector
import initializer_raw_marginal_experiment as experiment
import development_initializer_raw_marginal as runner
import initializer_target_label_control as original_control
from selector_training import selector_order


def design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(4)]
    return labels, folds


def raw_scores(x, y, permutation=None):
    """Direct independent global scaling and per-electrode linear solve."""
    sx, sy = x.std(axis=0), y.std(axis=0)
    sx[sx == 0], sy[sy == 0] = 1., 1.
    z, target = (x-x.mean(axis=0))/sx, (y-y.mean(axis=0))/sy
    if permutation is not None:
        target = target[permutation]
    scores = []
    for e in range(x.shape[1]//3):
        block = z[:, 3*e:3*e+3]
        cross = block.T@target/len(x)
        coefficient = np.linalg.solve(np.eye(3)+block.T@block/len(x), cross)
        scores.append(float(np.sum(cross*coefficient)/y.shape[1]))
    return np.asarray(scores)


class InitializerRawMarginalTest(unittest.TestCase):
    def test_global_scaling_and_scores_match_original_raw_formula_not_residual_coordinates(self):
        rng = np.random.default_rng(115001)
        labels, _ = design()
        channels, ids = [2, 7, 15, 30], np.arange(32)*11+100
        x = rng.normal(size=(32, 12))+labels[:, None]*np.arange(1, 13)[None, :]
        y = rng.normal(size=(32, 5))+labels[:, None]*np.arange(1, 6)[None, :]
        x[:, 4], y[:, 3] = 5., 7.
        before_x, before_y = x.copy(), y.copy()
        prepared = selector.prepare_view(x, y, labels, channels)
        sx, sy = x.std(axis=0), y.std(axis=0)
        sx[sx == 0], sy[sy == 0] = 1., 1.
        np.testing.assert_allclose(prepared["blocks"].reshape(x.shape), (x-x.mean(axis=0))/sx, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(prepared["yy"], (y-y.mean(axis=0))/sy, rtol=1e-14, atol=1e-14)
        self.assertEqual(list(prepared["zero_SD_input_columns"]), [4])
        self.assertEqual(list(prepared["zero_SD_target_columns"]), [3])
        residual_blocks = original_control.prepare_view(x, y, labels, channels)[0]
        self.assertFalse(np.allclose(prepared["blocks"], residual_blocks))
        order, scores, maps = selector.rank_prepared(prepared, labels, ids, channels, "example", "outer")
        np.testing.assert_allclose(scores, raw_scores(x, y), rtol=2e-13, atol=2e-14)
        np.testing.assert_array_equal(order, selector_order(x, y, labels, "marginal_target", len(channels)))
        self.assertEqual(maps, [])
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)
        tied, zero, _ = selector.rank_view(np.zeros_like(x), np.zeros_like(y), labels, ids,
                                          channels, "example", "outer")
        np.testing.assert_array_equal(tied, np.arange(4))
        np.testing.assert_array_equal(zero, np.zeros(4))

    def test_original_ID_maps_change_only_global_target_cross_products_and_allow_identity(self):
        rng = np.random.default_rng(115002)
        labels, _ = design()
        channels, ids = [2, 7, 15, 30], rng.permutation(np.arange(32)*17+100)
        x, y = rng.normal(size=(32, 12)), rng.normal(size=(32, 4))
        prepared = selector.prepare_view(x, y, labels, channels)
        frozen = {key: value.copy() for key, value in prepared.items() if isinstance(value, np.ndarray)}
        expected_map, expected_records = original_control.within_direction_map(ids, labels, "example", "outer", 3)
        order, gains, records = selector.rank_prepared(prepared, labels, ids, channels, "example", "outer", 3)
        self.assertEqual(records, expected_records)
        np.testing.assert_allclose(gains, raw_scores(x, y, expected_map), rtol=2e-13, atol=2e-14)
        reverse = np.arange(32)[::-1]
        _, _, reversed_records = selector.rank_view(x[reverse], y[reverse], labels[reverse], ids[reverse],
                                                    channels, "example", "outer", 3)
        self.assertEqual(records, reversed_records)
        for key, value in frozen.items():
            np.testing.assert_array_equal(prepared[key], value)
        literals = []
        class Identity:
            def seed(self, literal, version): literals.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(original_control.random, "Random", Identity):
            same_order, same_scores, _ = selector.rank_prepared(prepared, labels, ids, channels,
                                                               "example", "outer", 3)
        true_order, true_scores, _ = selector.rank_prepared(prepared, labels, ids, channels, "example", "outer")
        np.testing.assert_array_equal(same_order, true_order)
        np.testing.assert_array_equal(same_scores, true_scores)
        self.assertEqual(literals, [(f"ep08-acquired-marginal-target-label-v1|20261004|example|outer|3|{d}", 2)
                                     for d in range(8)])

    def test_nonfinite_normalization_fails_even_when_overflow_would_make_finite_zero_coordinates(self):
        labels, _ = design()
        for field in ("X", "Y"):
            x, y = np.zeros((32, 6)), np.zeros((32, 2))
            target = x if field == "X" else y
            target[:, 0] = np.tile([-1e200, 1e200], 16)
            with self.subTest(field=field), np.errstate(over="ignore", invalid="ignore"):
                with self.assertRaises(ArithmeticError):
                    selector.prepare_view(x, y, labels, [2, 9])

    def test_within_direction_controls_retain_direction_only_target_association(self):
        labels, _ = design()
        d = labels.astype(float)
        x = np.column_stack((d, d*d, np.ones(32), 7-d, 2*d, np.zeros(32)))
        y, ids, channels = np.column_stack((d, d*d)), np.arange(32)*7+10, [2, 9]
        prepared = selector.prepare_view(x, y, labels, channels)
        true_order, true_scores, _ = selector.rank_prepared(prepared, labels, ids, channels, "example", "outer")
        self.assertTrue(np.all(true_scores > 0))
        for seed in selector.MC_SEEDS:
            order, scores, _ = selector.rank_prepared(prepared, labels, ids, channels, "example", "outer", seed)
            np.testing.assert_array_equal(order, true_order)
            np.testing.assert_array_equal(scores, true_scores)
        residual = original_control.prepare_view(x, y, labels, channels)
        _, residual_scores, _ = original_control.rank_prepared(residual, labels, ids, channels, "example", "outer", 0)
        np.testing.assert_array_equal(residual_scores, np.zeros(2))

    def test_main_validation_exclusion_and_all30_orders_have_fixed_work_counts(self):
        rng = np.random.default_rng(115003)
        labels, folds = design()
        ids, channels = np.arange(32)*13+100, list(range(1, 97))
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 3))
        progress = defaultdict(int)
        banks, records = selector.order_bank(x, y, labels, ids, channels, folds, "example", progress)
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[folds[0][1]] += 1e6*rng.normal(size=(8, 288))
        altered_y[folds[0][1]] += 1e6*rng.normal(size=(8, 3))
        other, other_records = selector.order_bank(altered_x, altered_y, labels, ids, channels,
                                                   folds, "example", defaultdict(int))
        self.assertEqual(len(records), 30)
        for key, value in (("selector_orderings", 30), ("direction_permutation_maps", 200),
                           ("marginal_score_solves", 2880), ("raw_normalization_views", 5)):
            self.assertEqual(progress[key], value)
        for branch in selector.BRANCHES:
            np.testing.assert_array_equal(banks[branch]["0"], other[branch]["0"])
            first = next(r for r in records if r["mc_seed"] == branch and r["view"] == "0")
            second = next(r for r in other_records if r["mc_seed"] == branch and r["view"] == "0")
            self.assertEqual(first, second)
            for order in banks[branch].values():
                self.assertEqual(len(order), 32)
                self.assertEqual(len(set(order.tolist())), 32)

    def test_all30_orders_precede_future_reads_and95_distinct_heads_use_true_Y(self):
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
        rng = np.random.default_rng(115004)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 3))
        true_y, progress, calls = y.copy(), defaultdict(int), []
        case = self
        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
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
        np.testing.assert_array_equal(y, true_y)

    def test_scoring_averages_five_SSEs_and_keeps_shared_full96_cells(self):
        labels = np.repeat(np.arange(8), 5)
        x, y, channels = np.zeros((40, 288)), np.ones((40, 2)), list(range(1, 97))
        stats = SimpleNamespace(target_mean=np.zeros(2), target_sd=np.ones(2))
        null = SimpleNamespace(direction_target_mean=np.zeros((8, 2)),
                               target_mean=np.zeros(2), target_sd=np.ones(2))
        models = []
        for seed in selector.BRANCHES:
            for total in (32, 48, 64, 80, 96):
                for count in (8, 16, 32, 96):
                    value = 1. if seed is None or count == 96 else float(seed+1)
                    fitted = SimpleNamespace(statistics=stats, rank=1, alpha=1., candidate_count=25, value=value)
                    models.append((seed, total, count, tuple(channels[:count]), fitted))
        with patch.object(experiment, "predict_capacity", side_effect=lambda model, xx, dd:
                          np.full((len(xx), 2), model.value)):
            panels, seeds = experiment.score_target(models, {n: null for n in (32, 48, 64, 80, 96)},
                                                     x, y, labels, channels)
        self.assertEqual(len(seeds), 5)
        control = panels[selector.CONTROL]
        self.assertEqual(len(control["cells"]), 20)
        self.assertAlmostEqual(control["mean_Q"], -3.5)
        for cell in control["cells"]:
            self.assertFalse({"channels", "rank", "alpha", "candidate_count", "mc_seed"} & cell.keys())
            self.assertEqual(cell["null_SSE"], 80.)
            if cell["electrodes"] == 96:
                self.assertEqual(cell["Q"], 1.)
                self.assertEqual(cell["standardized_SSE"], 0.)
            else:
                # Mean losses [0,80,320,720,1280] = 480; averaging the
                # five predictions first would instead give SSE 320.
                self.assertEqual(cell["standardized_SSE"], 480.)
                self.assertEqual(cell["Q"], -5.)

    def test_reference_maps_legacy_schema_all_direction_nulls_and_exact_new_full96(self):
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
        labels, folds = design()
        views = {str(i): fit for i, (fit, _) in enumerate(folds)}
        views["outer"] = np.arange(32)
        ids = np.arange(32)*17+100
        sessions, old = [], []
        for filename in runner.DEVELOPMENT:
            support = {"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                       "source_roles": {"filename": filename}, "target_unit_count": 10}
            records = [{"mc_seed": seed, "view": view,
                        "target_label_permutations": original_control.within_direction_map(
                            ids[fit], labels[fit], filename, view, seed)[1]}
                       for seed in selector.MC_SEEDS for view, fit in views.items()]
            marginal = panel(.4)
            for cell in [*marginal["cells"], *marginal["fitted_models"]]:
                for key in ("null_SSE", "standardized_SSE", "standardized_wrong_SSE"):
                    cell.pop(key)
            old.append({**support, "selection_records": records,
                "panels": {"initial_marginal": marginal, experiment.PREVIOUS_CONTROL: panel(.2)},
                "seed_panels": [{"mc_seed": seed, "panel": panel(.2)} for seed in selector.MC_SEEDS]})
            sessions.append({**support, "selection_records": [*copy.deepcopy(records),
                *[{"mc_seed": None, "view": view, "target_label_permutations": []} for view in views]],
                "panels": {selector.POLICY: panel(.35), selector.CONTROL: panel(.25)},
                "seed_panels": [{"mc_seed": seed, "panel": panel(.25)} for seed in selector.MC_SEEDS]})
        previous = {"status": "completed_one_initializer_target_label_control",
            "recipe": "outputs/initializer_target_label_control_proposal.md", "held_session_accessed": False,
            "control": experiment.PREVIOUS_CONTROL, "MC_seeds": list(selector.MC_SEEDS), "sessions": old}
        accepted = copy.deepcopy(sessions)
        self.assertEqual(experiment.attach_references(accepted, previous), (0., 0.))
        self.assertEqual(set(accepted[0]["panels"]), set(experiment.ALL_POLICIES))
        self.assertNotIn("null_SSE", accepted[0]["panels"]["initial_marginal"]["cells"][0])
        for kind in ("map", "required_SSE", "partial_legacy", "legacy_D", "seed_D", "old_capacity"):
            changed = copy.deepcopy(previous)
            reference = changed["sessions"][0]
            if kind == "map":
                reference["selection_records"][0]["target_label_permutations"][0]["donor_trial_ids"][0] = -1
            elif kind == "required_SSE":
                reference["panels"][experiment.PREVIOUS_CONTROL]["cells"][0].pop("null_SSE")
            elif kind == "partial_legacy":
                reference["panels"]["initial_marginal"]["cells"][0]["null_SSE"] = 100.
            elif kind == "legacy_D":
                reference["panels"]["initial_marginal"]["cells"][0]["direction_only_Q"] += .01
            elif kind == "seed_D":
                reference["seed_panels"][4]["panel"]["cells"][0]["direction_only_Q"] += .01
            elif kind == "old_capacity":
                next(r for r in reference["seed_panels"][0]["panel"]["fitted_models"]
                     if r["electrodes"] == 96)["alpha"] = 100.
            with self.subTest(kind=kind), self.assertRaises((ValueError, ArithmeticError)):
                experiment.attach_references(copy.deepcopy(sessions), changed)
        for field, value in (("Q", .3+5e-12), ("rank", 2), ("null_SSE", 101.)):
            changed = copy.deepcopy(sessions)
            next(r for r in changed[0]["seed_panels"][4]["panel"]["fitted_models"]
                 if r["electrodes"] == 96)[field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, ArithmeticError)):
                experiment.attach_references(changed, previous)

    def test_missing_initial_model_or_order_blocks_evaluation(self):
        for model_count, order_count in ((94, 30), (95, 29)):
            with self.subTest(models=model_count, orders=order_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata, roles = root/"metadata.json", root/"roles.json"
                metadata.write_text("{}")
                roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
                args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                    output_dir=root/"attempt", max_seconds=270, reference_report=root/"closed_reference.json")
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

    def test_all760_new_models_precede_original_reference_read(self):
        class ReferenceReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                output_dir=root/"attempt", max_seconds=270, reference_report=root/"unread_reference.json")
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
                         "evaluation_trials": list(range(40))},
                        {i: {"direction_index": i//5} for i in range(40)})
            def fit(path, source, role, records, progress, check, exposure):
                self.assertNotIn(args.reference_report, read_paths)
                fitted.append(path.name)
                for key, value in (("outer_training_refits", 95), ("inner_ridge_fits", 1900),
                        ("capacity_inner_candidate_scores", 9500), ("selector_orderings", 30),
                        ("marginal_score_solves", 2880), ("raw_normalization_views", 5),
                        ("direction_permutation_maps", 200), ("direction_only_models", 5)):
                    progress[key] += value
                exposure.report["fitting_trial_read_calls"] += 1248
                return [], {}, None, []
            def evaluation(path, source, role, records, ids, channels, check, exposure, guide, evaluation):
                self.assertTrue(evaluation)
                self.assertEqual(fitted[-1], path.name)
                evaluated.append(path.name)
                exposure.report["evaluation_trials_read"] += 40
                return np.zeros((40, 288)), np.ones((40, 3)), None
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "selected_metadata", side_effect=metadata_source), \
                    patch.object(runner, "fit_target", side_effect=fit), \
                    patch.object(runner, "read_trials", side_effect=evaluation), \
                    patch.object(runner, "score_target", return_value=({}, [])), \
                    patch.object(runner, "role_record", side_effect=lambda source, role: {"filename": role["filename"]}), \
                    patch.object(Path, "read_text", read_text):
                with self.assertRaises(ReferenceReached):
                    runner.main()
            self.assertEqual(len(fitted), 8)


if __name__ == "__main__":
    unittest.main()
