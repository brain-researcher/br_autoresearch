"""Independent internal-validation and control checks for utility-based selection."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import initializer_cv_utility as selector
import initializer_cv_utility_experiment as experiment
import development_initializer_cv_utility as runner


def main_design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(4)]
    return labels, folds


def direct_utility(x, y, labels, fit, valid, permutation=None):
    """Independent raw-array normalization and normalized ridge validation loss."""
    xf, yf = x[fit], y[fit]
    fit_labels, val_labels = labels[fit], labels[valid]
    mx = np.stack([xf[fit_labels == d].mean(axis=0) for d in range(8)])
    my = np.stack([yf[fit_labels == d].mean(axis=0) for d in range(8)])
    dx, dy = xf-mx[fit_labels], yf-my[fit_labels]
    cx, sx, sy = dx.mean(axis=0), dx.std(axis=0), yf.std(axis=0)
    sx[sx == 0], sy[sy == 0] = 1., 1.
    zf, zv = (dx-cx)/sx, (x[valid]-mx[val_labels]-cx)/sx
    b = (dy/sy).mean(axis=0)
    rf, r0 = dy/sy-b, (y[valid]-my[val_labels])/sy
    if permutation is not None:
        rf = rf[permutation]
    scores = []
    for electrode in range(x.shape[1]//3):
        block = slice(3*electrode, 3*electrode+3)
        a = zf[:, block]
        coefficient = np.linalg.solve(a.T@a+len(fit)*np.eye(3), a.T@rf)
        corrected = r0-b-zv[:, block]@coefficient
        scores.append(float(np.sum(r0**2-corrected**2)))
    return np.asarray(scores)


class InitializerCVUtilityTest(unittest.TestCase):
    def test_internal_folds_use_only_original_ID_direction_metadata(self):
        for m in (3, 4):
            labels = np.repeat(np.arange(8), m)
            ids = np.random.default_rng(98000+m).permutation(np.arange(8*m)*13+100)
            folds = selector.internal_folds(ids, labels)
            self.assertEqual(len(folds), m)
            validated = []
            for fit, valid in folds:
                self.assertEqual(len(fit), 8*(m-1))
                self.assertEqual(len(valid), 8)
                self.assertFalse(set(fit) & set(valid))
                self.assertEqual(set(fit) | set(valid), set(range(8*m)))
                np.testing.assert_array_equal(np.bincount(labels[fit], minlength=8), np.full(8, m-1))
                np.testing.assert_array_equal(np.bincount(labels[valid], minlength=8), np.ones(8, dtype=int))
                validated.extend(valid.tolist())
            self.assertEqual(sorted(validated), list(range(8*m)))
            reverse = np.arange(8*m)[::-1]
            other = selector.internal_folds(ids[reverse], labels[reverse])
            for (fit, valid), (other_fit, other_valid) in zip(folds, other):
                self.assertEqual(set(ids[fit]), set(ids[reverse][other_fit]))
                self.assertEqual(set(ids[valid]), set(ids[reverse][other_valid]))

    def test_internal_fit_state_excludes_validation_X_Y_and_matches_direct_loss(self):
        rng = np.random.default_rng(98005)
        labels = np.repeat(np.arange(8), 3)
        ids = np.arange(24)*11+100
        x, y = rng.normal(size=(24, 12)), rng.normal(size=(24, 5))
        x[:, 7], y[:, 3] = 0., 4.
        fit, valid = selector.internal_folds(ids, labels)[1]
        prepared = selector.prepare_internal_fold(x, y, labels, fit, valid)
        np.testing.assert_allclose(selector.validation_utilities(prepared),
                                   direct_utility(x, y, labels, fit, valid), rtol=2e-12, atol=2e-12)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[valid] += 100*rng.normal(size=(8, 12))
        changed_y[valid] -= 100*rng.normal(size=(8, 5))
        poisoned = selector.prepare_internal_fold(changed_x, changed_y, labels, fit, valid)
        for key in ("xx", "yy", "b", "gram"):
            np.testing.assert_array_equal(prepared[key], poisoned[key])
        self.assertFalse(np.array_equal(prepared["xv"], poisoned["xv"]))
        self.assertFalse(np.array_equal(prepared["r0v"], poisoned["r0v"]))

    def test_signed_validation_utility_retains_negative_and_zero_values(self):
        labels = np.repeat(np.arange(8), 3)
        ids = np.arange(24)
        fit, valid = selector.internal_folds(ids, labels)[0]
        x, y = np.zeros((24, 6)), np.zeros((24, 1))
        for direction in range(8):
            rows = fit[labels[fit] == direction]
            x[rows, 0], x[rows, 3], y[rows, 0] = [-1., 1.], [-1., 1.], [-1., 1.]
        x[valid, 0], x[valid, 3], y[valid, 0] = 1., 1., -1.
        prepared = selector.prepare_internal_fold(x, y, labels, fit, valid)
        np.testing.assert_allclose(selector.validation_utilities(prepared), [-10., -10.], atol=2e-13)
        zero = selector.prepare_internal_fold(np.zeros_like(x), y, labels, fit, valid)
        np.testing.assert_array_equal(selector.validation_utilities(zero), np.zeros(2))

    def test_control_maps_stay_inside_F_preserve_direction_and_accept_identity(self):
        rng = np.random.default_rng(98006)
        labels, _ = main_design()
        ids = np.arange(32)*17+200
        x, y = rng.normal(size=(32, 12)), rng.normal(size=(32, 4))
        fit, valid = selector.internal_folds(ids, labels)[2]
        prepared = selector.prepare_internal_fold(x, y, labels, fit, valid)
        mapping, records = selector.fit_only_map(ids[fit], labels[fit], "example", "outer", 2, 4)
        np.testing.assert_array_equal(labels[fit][mapping], labels[fit])
        self.assertEqual(set(mapping.tolist()), set(range(24)))
        self.assertFalse(set(ids[fit][mapping]) & set(ids[valid]))
        self.assertEqual(len(records), 8)
        np.testing.assert_allclose(selector.validation_utilities(prepared, mapping),
            direct_utility(x, y, labels, fit, valid, mapping), rtol=2e-12, atol=2e-12)
        draws = []
        class Identity:
            def seed(self, literal, version): draws.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(selector.random, "Random", Identity):
            same, same_records = selector.fit_only_map(ids[fit], labels[fit], "example", "outer", 2, 4)
        np.testing.assert_array_equal(same, np.arange(24))
        np.testing.assert_array_equal(selector.validation_utilities(prepared, same), selector.validation_utilities(prepared))
        self.assertEqual(draws, [(f"ep08-initial-cv-utility-fit-label-v1|20261004|example|outer|2|4|{d}", 2)
                                 for d in range(8)])
        self.assertEqual(len(same_records), 8)

    def test_rank_aggregates_signed_validation_loss_and_uses_physical_ties(self):
        rng = np.random.default_rng(98007)
        labels = np.repeat(np.arange(8), 3)
        ids, channels = np.arange(24)*19+30, [2, 9, 20, 31]
        x, y = rng.normal(size=(24, 12)), rng.normal(size=(24, 3))
        prepared = selector.prepare_view(x, y, labels, ids, channels)
        progress = defaultdict(int)
        order, gains, records = selector.rank_prepared(prepared, ids, labels, channels,
                                                       "example", "0", None, progress)
        expected = sum(direct_utility(x, y, labels, f, v)
                       for f, v in selector.internal_folds(ids, labels))/(24*3)
        np.testing.assert_allclose(gains, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(order, np.lexsort((channels, -expected)))
        self.assertEqual(progress["proxy_ridge_models"], 12)
        self.assertEqual(progress["direction_permutation_maps"], 0)
        self.assertEqual(len(records), 3)
        zero = selector.prepare_view(np.zeros_like(x), np.zeros_like(y), labels, ids, channels)
        tied, scores, _ = selector.rank_prepared(zero, ids, labels, channels,
                                                "example", "0", None, defaultdict(int))
        np.testing.assert_array_equal(tied, np.arange(4))
        np.testing.assert_array_equal(scores, np.zeros(4))

    def test_all30_orders_preserve_main_held_exclusion_and_count_internal_work(self):
        rng = np.random.default_rng(98008)
        labels, folds = main_design()
        ids, channels = np.arange(32)*23+500, list(range(1, 97))
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 3))
        progress = defaultdict(int)
        orders, records = selector.order_bank(x, y, labels, ids, channels, folds, "example", progress)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += 1e6*rng.normal(size=(8, 288))
        changed_y[folds[0][1]] += 1e6*rng.normal(size=(8, 3))
        other, other_records = selector.order_bank(changed_x, changed_y, labels, ids, channels,
                                                   folds, "example", defaultdict(int))
        self.assertEqual(len(records), 30)
        for key, value in (("proxy_ridge_models", 9216), ("proxy_normalization_views", 16),
                           ("direction_permutation_maps", 640), ("selector_orderings", 30)):
            self.assertEqual(progress[key], value)
        for seed in selector.BRANCHES:
            np.testing.assert_array_equal(orders[seed]["0"], other[seed]["0"])
            left = next(r for r in records if r["view"] == "0" and r["mc_seed"] == seed)
            right = next(r for r in other_records if r["view"] == "0" and r["mc_seed"] == seed)
            self.assertEqual(left, right)
            for order in orders[seed].values():
                self.assertEqual(len(order), 32)
                self.assertEqual(len(set(order.tolist())), 32)
        for record in records:
            main_ids = set(record["fitting_trial_ids"])
            for internal in record["internal_folds"]:
                fitting = set(internal["fitting_trial_ids"])
                validation = set(internal["validation_trial_ids"])
                self.assertEqual(fitting | validation, main_ids)
                self.assertFalse(fitting & validation)
                for mapping in internal["target_label_permutations"]:
                    self.assertTrue(set(mapping["donor_trial_ids"]) <= fitting)
                    self.assertEqual(set(mapping["donor_trial_ids"]), set(mapping["recipient_trial_ids"]))

    def test_all30_orders_precede_future_reads_and95_distinct_heads_use_true_Y(self):
        labels, folds = main_design()
        filename = "Mihili_CO_20140217.mat"
        pools = [list(range(12*d, 12*d+12)) for d in range(8)]
        ids = [trial for pool in pools for trial in pool[:4]]
        role = {"filename": filename, "animal": "Mihili", "fitting_pool_by_direction": pools,
                "nested_training_sets": {"32": {"training_trials": ids, "inner_folds": [
                    {"fitting_trials": [ids[i] for i in fit], "validation_trials": [ids[i] for i in valid]}
                    for fit, valid in folds]}}}
        records = {trial: {"direction_index": d} for d, pool in enumerate(pools) for trial in pool}
        source = {"common_physical_channels": list(range(1, 97))}
        rng = np.random.default_rng(98009)
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
        for total in (32, 48, 64, 80, 96):
            self.assertEqual(len({id(row[-1]) for row in models if row[1:3] == (total, 96)}), 1)

    def test_reference_replay_accepts_legacy_absence_but_checks_all_new_control_full96(self):
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
                "seed_panels": [{"mc_seed": seed, "panel": panel(.25)} for seed in selector.MC_SEEDS]})
            sessions.append({**support, "panels": {selector.POLICY: panel(.35), selector.CONTROL: panel(.2)},
                "seed_panels": [{"mc_seed": seed, "panel": panel(.2)} for seed in selector.MC_SEEDS]})
        reference = {"status": "completed_two_initializer_repeatability_comparisons",
            "held_session_accessed": False, "policy": experiment.REPEATABILITY,
            "control": experiment.UNIFORM, "MC_seeds": list(selector.MC_SEEDS),
            "recipe": "outputs/initializer_repeatability_proposal.md", "sessions": previous}
        accepted = copy.deepcopy(sessions)
        self.assertEqual(experiment.attach_references(accepted, reference), (0., 0.))
        self.assertEqual(set(accepted[0]["panels"]), set(experiment.ALL_POLICIES))
        self.assertNotIn("null_SSE", accepted[0]["panels"]["initial_marginal"]["cells"][0])
        for branch in (None, 0, 4):
            for key, value in (("rank", 2), ("Q", .31), ("null_SSE", 101.), ("LFP_numeric_elements", 999)):
                changed = copy.deepcopy(sessions)
                target = (changed[0]["panels"][selector.POLICY] if branch is None else
                          changed[0]["seed_panels"][branch]["panel"])
                row = next(r for r in target["fitted_models"] if r["electrodes"] == 96)
                row[key] = value
                with self.subTest(branch=branch, field=key), self.assertRaises((ValueError, ArithmeticError)):
                    experiment.attach_references(changed, reference)
        changed = copy.deepcopy(reference)
        changed["sessions"][0]["panels"][experiment.UNIFORM]["cells"][0].pop("null_SSE")
        with self.assertRaises(ValueError):
            experiment.attach_references(copy.deepcopy(sessions), changed)

    def test_pilot_reuse_requires_exact_recipe_roles_all_internal_work_and_zero_charge(self):
        expected = {"filename": runner.PILOT, "initial_trial_ids": list(range(32))}
        session = {"filename": runner.PILOT, "animal": "Mihili", "source_roles": copy.deepcopy(expected),
            "all95_target_models_fixed_before_evaluation": True,
            "all30_target_orders_fixed_before_post_initial_reads": True}
        pilot = {"status": "completed_initializer_cv_utility_pilot_scores_withheld", "mode": "pilot",
            "recipe": runner.RECIPE, "policy": selector.POLICY, "control": selector.CONTROL,
            "MC_seeds": list(selector.MC_SEEDS), "held_session_accessed": False,
            "scientific_scores_released": False, "valid_evaluations_charged": 0,
            "completed_session_ids": [runner.PILOT], "sessions": [session],
            "fitting_trial_read_calls": 1248, "evaluation_trials_read": 40,
            "fit_progress": {"outer_training_refits": 95, "inner_ridge_fits": 1900,
                "capacity_inner_candidate_scores": 9500, "selector_orderings": 30,
                "proxy_ridge_models": 9216, "proxy_normalization_views": 16,
                "direction_permutation_maps": 640, "direction_only_models": 5}}
        sources = {runner.PILOT: ({}, {"animal": "Mihili"}, {})}
        with patch.object(runner, "role_record", return_value=expected):
            self.assertIs(runner.validate_pilot(pilot, sources), session)
            for key, value in (("recipe", "other"), ("valid_evaluations_charged", 1),
                               ("scientific_scores_released", True), ("evaluation_trials_read", 39)):
                changed = copy.deepcopy(pilot)
                changed[key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    runner.validate_pilot(changed, sources)
            for key in pilot["fit_progress"]:
                changed = copy.deepcopy(pilot)
                changed["fit_progress"][key] -= 1
                with self.subTest(progress=key), self.assertRaises(ValueError):
                    runner.validate_pilot(changed, sources)
            changed = copy.deepcopy(pilot)
            changed["sessions"][0]["source_roles"]["initial_trial_ids"][0] = 999
            with self.assertRaises(ValueError):
                runner.validate_pilot(changed, sources)

    def test_missing_initial_model_or_order_blocks_evaluation(self):
        for model_count, order_count in ((94, 30), (95, 29)):
            with self.subTest(models=model_count, orders=order_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata, roles = root/"metadata.json", root/"roles.json"
                metadata.write_text("{}")
                roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
                args = SimpleNamespace(mode="pilot", source_root=root, metadata_report=metadata, trial_roles=roles,
                    output_dir=root/"attempt", max_seconds=270, pilot_report=None, reference_report=None)
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

    def test_remaining_seven_finish_before_pilot_and_historical_score_reads(self):
        class DeferredPilotObserved(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(mode="remaining", source_root=root, metadata_report=metadata, trial_roles=roles,
                output_dir=root/"attempt", max_seconds=270, pilot_report=root/"unread_pilot.json",
                reference_report=root/"unread_reference.json")
            fitted, evaluated, read_paths = [], [], []
            original_read = Path.read_text
            def read_text(path, *pargs, **kwargs):
                read_paths.append(path)
                if path in (args.pilot_report, args.reference_report):
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
                self.assertFalse(any(p in read_paths for p in (args.pilot_report, args.reference_report)))
                fitted.append(path.name)
                for key, value in (("outer_training_refits", 95), ("inner_ridge_fits", 1900),
                        ("capacity_inner_candidate_scores", 9500), ("selector_orderings", 30),
                        ("proxy_ridge_models", 9216), ("proxy_normalization_views", 16),
                        ("direction_permutation_maps", 640), ("direction_only_models", 5)):
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
                with self.assertRaises(DeferredPilotObserved):
                    runner.main()
            self.assertEqual(len(fitted), 7)
            self.assertNotIn(args.reference_report, read_paths)


if __name__ == "__main__":
    unittest.main()
