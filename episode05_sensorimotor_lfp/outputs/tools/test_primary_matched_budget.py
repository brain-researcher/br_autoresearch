"""Independent support, reduced-HW, reference and accounting fixtures."""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np

import primary_history as history
import primary_history_repeatability_weight as hw
import primary_matched_budget as matched
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(dtype=np.float64, seed=105101):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 4)
    # Mimic the qualified row-by-time reader before stacking whole trials.
    source = [rng.normal(size=(5, 18)).astype(dtype) for _ in labels]
    x = np.stack([row.T.astype(np.float64) for row in source])
    x += np.sin(np.linspace(-2., 2., 18))[None, :, None] * np.arange(1., 6.)
    y = rng.normal(size=(len(labels), 15, 2)) + .3*x[:, 3:, :2]
    ids = np.array([f"example_trial{i+1:06d}" for i in range(len(labels))])
    return x, y, labels, ids, [2, 4, 7, 11, 19]


def ordinary(values, literal):
    values = list(values)
    generator = random.Random()
    generator.seed(literal, version=2)
    generator.shuffle(values)
    return values


class IdentityShuffle:
    def seed(self, literal, version):
        if version != 2:
            raise AssertionError("The declared ordinary string-seed version is two")

    def shuffle(self, values):
        pass


class MatchedBudgetTest(unittest.TestCase):
    def test_global_metadata_reconciles_all_sessions_and_freezes_both_minima(self):
        records, rows = {}, []
        for session in SESSIONS:
            counts = [13]*8
            if session == "Chewie_20150629":
                counts[3] = 12
            labels = np.repeat(np.arange(8), counts)
            records[session] = [{"trial_id": f"{session}_trial{i+1:06d}", "direction": int(d)}
                                for i, d in enumerate(labels)]
            electrodes = list(range(1, (86 if session in ("Chewie_20150313", "Chewie_20150319") else 96)+1))
            rows.append({"session_id": session, "physical_LMP_electrode_count": len(electrodes),
                         "physical_LMP_electrode_ids": electrodes,
                         "direction_counts": {str(d): n for d, n in enumerate(counts)},
                         "complete_trials": len(labels)})
        metadata = {"sessions": rows, "source_values_read": False, "score_values_released": False,
                    "minimum_electrode_count": 86, "minimum_direction_trial_count": 12}
        result = matched.validate_global_metadata(metadata, records)
        self.assertEqual(set(result), set(SESSIONS))
        for supports in result.values():
            for recipe in (matched.RECIPES[1], matched.RECIPES[2]):
                self.assertTrue(all(len(row["trial_ids"]) == 96 for row in supports[recipe]))
            for recipe in (matched.RECIPES[0], matched.RECIPES[2]):
                self.assertTrue(all(len(row["electrode_ids"]) == 86 for row in supports[recipe]))
        for change in ("missing_session", "changed_budget", "wrong_direction_counts", "duplicate_id", "values_read",
                       "integer_keys", "missing_direction", "extra_direction", "noninteger_count", "bool_count",
                       "wrong_global_minimum", "legacy_list"):
            other, other_records = copy.deepcopy(metadata), copy.deepcopy(records)
            if change == "missing_session":
                other["sessions"].pop()
            elif change == "changed_budget":
                other["minimum_electrode_count"] = 85
            elif change == "wrong_direction_counts":
                other["sessions"][0]["direction_counts"]["0"] += 1
            elif change == "duplicate_id":
                other_records[SESSIONS[0]][1]["trial_id"] = other_records[SESSIONS[0]][0]["trial_id"]
            elif change == "values_read":
                other["source_values_read"] = True
            elif change == "integer_keys":
                other["sessions"][0]["direction_counts"] = {int(k): v for k, v in rows[0]["direction_counts"].items()}
            elif change == "missing_direction":
                del other["sessions"][0]["direction_counts"]["7"]
            elif change == "extra_direction":
                other["sessions"][0]["direction_counts"]["8"] = 13
            elif change == "noninteger_count":
                other["sessions"][0]["direction_counts"]["0"] = 13.5
            elif change == "bool_count":
                other["sessions"][0]["direction_counts"]["0"] = True
            elif change == "legacy_list":
                other["sessions"][0]["direction_counts"] = list(rows[0]["direction_counts"].values())
            else:
                for row in other["sessions"]:
                    row["direction_counts"] = {str(d): 13 for d in range(8)}
            with self.assertRaises((ValueError, ArithmeticError)):
                matched.validate_global_metadata(other, other_records)

    def test_literal_coupled_ordinary_queues_restore_original_support_order(self):
        _, _, labels, ids, electrodes = example()
        supports = matched.support_views(ids, labels, electrodes, "example",
                                         electrode_budget=3, trials_per_direction=3)
        self.assertEqual(list(supports), list(matched.RECIPES))
        for view in range(8):
            channels = sorted(ordinary(electrodes,
                f"ep05-HW-budget-electrode-v1|20261004|example|{view}")[:3])
            selected_ids = []
            for direction in range(8):
                original = ids[labels == direction].tolist()
                selected_ids.extend(ordinary(original,
                    f"ep05-HW-budget-trial-v1|20261004|example|{view}|{direction}")[:3])
            selected_ids.sort()
            c, t, ct = [supports[recipe][view] for recipe in matched.RECIPES]
            for item, expected_ids, expected_channels in (
                    (c, ids.tolist(), channels), (t, selected_ids, electrodes),
                    (ct, selected_ids, channels)):
                self.assertEqual(item["view"], view)
                self.assertEqual(item["trial_ids"], expected_ids)
                self.assertEqual(item["electrode_ids"], expected_channels)
                np.testing.assert_array_equal(ids[item["trial_positions"]], expected_ids)
                np.testing.assert_array_equal(np.asarray(electrodes)[item["electrode_positions"]], expected_channels)
                self.assertFalse(item["full_support"])
            self.assertEqual(c["electrode_ids"], ct["electrode_ids"])
            self.assertEqual(t["trial_ids"], ct["trial_ids"])
        with patch.object(matched.random, "Random", IdentityShuffle):
            repeated = matched.support_views(ids, labels, electrodes, "example",
                                             electrode_budget=5, trials_per_direction=4)
        self.assertTrue(all(row["full_support"] for rows in repeated.values() for row in rows))
        for bad_ids, bad_labels, bad_channels, channel_count, trial_count in (
                (ids, labels, electrodes, 6, 3), (ids, labels, electrodes, 3, 5),
                (np.array([*ids[:-1], ids[0]]), labels, electrodes, 3, 3),
                (np.array([value.replace("example_", "other_") for value in ids]), labels, electrodes, 3, 3),
                (ids, labels, [2, 4, 7, 11, 11], 3, 3)):
            with self.assertRaises((ValueError, ArithmeticError)):
                matched.support_views(bad_ids, bad_labels, bad_channels, "example",
                    electrode_budget=channel_count, trials_per_direction=trial_count)

    def test_reduced_true_HW_matches_independent_augmented_ridge_on_reader_layout(self):
        for dtype in (np.float32, np.float64):
            x, y, labels, ids, channels = example(dtype)
            supports = matched.support_views(ids, labels, channels, "example",
                                             electrode_budget=3, trials_per_direction=3)
            support = supports[matched.RECIPES[2]][2]
            sx, sy, sd, si = matched.subset_arrays(x, y, labels, ids, support)
            expected_x = x[support["trial_positions"]][:, :, support["electrode_positions"]]
            np.testing.assert_array_equal(sx, expected_x)
            self.assertEqual(sx.strides, expected_x.strides)
            np.testing.assert_array_equal(si, support["trial_ids"])
            baseline, predictions, metadata = matched.fit_view(sx, sy, sd)
            self.assertEqual(predictions.shape, (5, len(sy), 15, 2))
            self.assertEqual(len(metadata), 8)
            blocks = history.history_blocks(sx)
            for held in range(8):
                split = hw.prepare_fold(blocks, sx[:, 3:18, :], sy, sd, held)
                for key in ("physical_mean", "physical_sd"):
                    np.testing.assert_array_equal(metadata[held][key], split["weight_state"][key])
                for key in ("T", "P", "r", "a"):
                    np.testing.assert_array_equal(metadata[held]["repeatability"][key], split["repeatability"][key])
                fitting = split["fitting_history"].reshape(-1, 12)
                evaluation = split["evaluation_history"].reshape(-1, 12)
                gram = fitting.T@fitting/len(fitting)
                a = split["repeatability"]["a"]
                q = np.diag(gram).reshape(4, 3).sum(axis=0)
                weights = np.sqrt(a*np.trace(gram)/(a@q))
                transformed = fitting*np.tile(weights, 4)
                test = evaluation*np.tile(weights, 4)
                base = split["template"].predict(sd[split["test"]]*np.pi/4)+split["intercept"]
                np.testing.assert_array_equal(baseline[split["test"]], base)
                for index, penalty in enumerate(LAMBDAS):
                    design = np.vstack((transformed, np.sqrt(len(fitting)*penalty)*np.eye(12)))
                    target = np.vstack((split["residual"], np.zeros((12, 2))))
                    beta = np.linalg.lstsq(design, target, rcond=None)[0]
                    expected = base+(test@beta).reshape(base.shape)
                    np.testing.assert_allclose(predictions[index, split["test"]], expected,
                                               rtol=2e-10, atol=2e-10)
            # Held Y is unavailable to both the residual target and baseline.
            changed = sy.copy()
            changed[sd == 4] += 1e5
            other_base, other_predictions, _ = matched.fit_view(sx, changed, sd)
            np.testing.assert_array_equal(other_base[sd == 4], baseline[sd == 4])
            np.testing.assert_array_equal(other_predictions[:, sd == 4], predictions[:, sd == 4])

    def test_exact_support_reuse_skips_full_and_repeated_condition_views(self):
        x, y, labels, ids, channels = example()
        with patch.object(matched.random, "Random", IdentityShuffle):
            supports = matched.support_views(ids, labels, channels, "example",
                                             electrode_budget=5, trials_per_direction=3)
        observed = []
        def fake_fit(features, targets, directions, check=lambda: None):
            observed.append((features.copy(), targets.copy(), directions.copy()))
            return np.zeros_like(targets), np.zeros((5, *targets.shape)), []
        with patch.object(matched, "fit_view", side_effect=fake_fit):
            matched.fit_reduced(x, y, labels, ids, supports, lambda: None)
        # C is entirely the saved full reference; every T and CT view has
        # exactly one common reduced support under the allowed identity shuffle.
        self.assertEqual(len(observed), 1)
        chosen = supports[matched.RECIPES[1]][0]
        expected = matched.subset_arrays(x, y, labels, ids, chosen)
        for actual, value in zip(observed[0], expected[:3]):
            np.testing.assert_array_equal(actual, value)
        removed = np.ones(len(y), dtype=bool)
        removed[chosen["trial_positions"]] = False
        changed_x, changed_y = x.copy(order="K"), y.copy()
        changed_x[removed] = 1e8
        changed_y[removed] = -1e8
        observed.clear()
        with patch.object(matched, "fit_view", side_effect=fake_fit):
            matched.fit_reduced(changed_x, changed_y, labels, ids, supports, lambda: None)
        self.assertEqual(len(observed), 1)
        for actual, value in zip(observed[0], expected[:3]):
            np.testing.assert_array_equal(actual, value)

    def test_mean_view_normalized_losses_and_exact_retained_trial_wrong_pairs(self):
        _, y, labels, _, _ = example()
        views = []
        for view in range(8):
            actual = (view+1)*y
            baseline = np.zeros_like(actual)+.2*view
            predictions = np.stack([actual*(.12+.1*i)+.3*view for i in range(5)])
            bank = score_bank(actual, labels, baseline, predictions, matched.RECIPES[1])
            for index in range(5):
                # Enumerate recipient/donor pairs directly, independently of
                # the production algebra for the mean over derangements.
                wrong = 0.
                for direction in range(8):
                    positions = np.flatnonzero(labels == direction)
                    wrong += sum(float(np.square(actual[i]-predictions[index, j]).sum())
                                 for i in positions for j in positions if i != j)/(len(positions)-1)
                self.assertAlmostEqual(bank["lambda_rows"][index]["wrong_pair_SSE"], wrong, places=7)
            views.append({"view": view, "bank": bank})
        result = matched.average_views(views, matched.RECIPES[1])
        denominators = np.array([v["bank"]["denominator"] for v in views])
        self.assertGreater(denominators.max()/denominators.min(), 60.)
        for index in range(5):
            p = np.array([v["bank"]["lambda_rows"][index]["candidate_SSE"] for v in views])
            q = np.array([v["bank"]["lambda_rows"][index]["wrong_pair_SSE"] for v in views])
            b = np.array([v["bank"]["baseline_SSE"] for v in views])
            row = result["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_R2"], 1-np.mean(p/denominators), places=13)
            self.assertAlmostEqual(row["delta_R2"], np.mean((b-p)/denominators), places=13)
            self.assertAlmostEqual(row["identity_increment"], np.mean((q-p)/denominators), places=13)
            self.assertGreater(abs(row["delta_R2"]-(b-p).mean()/denominators.mean()), 1e-5)
        self.assertNotIn("denominator", result)
        broken = copy.deepcopy(views)
        broken[0]["bank"]["denominator"] = 0.
        with self.assertRaises(ValueError):
            matched.average_views(broken, matched.RECIPES[1])

    def test_reference_ID_alignment_and_direct_loss_baseline_decomposition(self):
        x, y, labels, ids, channels = example()
        supports = matched.support_views(ids, labels, channels, "example",
                                         electrode_budget=3, trials_per_direction=3)
        def synthetic_fit(features, targets, directions, check=lambda: None):
            baseline = .13*targets+2.
            prediction = np.stack([(.15+.12*i)*targets+.4 for i in range(5)])
            metadata = [{"held_direction": d, "fitting_bin_rows": 15*int(np.sum(directions != d)),
                         "paired_weighted_covariance": {"trace": 12., "numerical_rank": 12,
                         "ridge_df_by_lambda": [{"lambda": p, "effective_df": 10.-i}
                                                for i, p in enumerate(LAMBDAS)]}} for d in range(8)]
            return baseline, prediction, metadata
        with patch.object(matched, "fit_view", side_effect=synthetic_fit):
            fits, routes = matched.fit_reduced(x, y, labels, ids, supports)
        full_base = -.1*y-.5
        full_pred = np.stack([(.75-.1*i)*y-.3 for i in range(5)])
        order = np.arange(len(y))[::-1]
        reference = {"trial_ids": ids[order], "directions": labels[order], "actual_xy": y[order],
                     "baseline_xy": full_base[order], "candidate_xy_by_lambda": full_pred[:, order],
                     "fold_metadata": synthetic_fit(x, y, labels)[2]}
        aligned = matched.align_reference(reference, y, labels, ids)
        np.testing.assert_array_equal(aligned["candidate_xy_by_lambda"], full_pred)
        banks, payload = matched.finalize_banks(y, labels, ids, supports, fits, routes, reference)
        self.assertTrue(payload)
        for bank in banks.values():
            for view in bank["views"]:
                selected = view["support"]["trial_positions"]
                actual = y[selected]
                denominator = float(np.square(actual-actual.mean(axis=(0, 1), keepdims=True)).sum())
                reduced = fits[view["fit_key"]]
                b = float(np.square(actual-reduced["baseline"]).sum())
                old_b = float(np.square(actual-full_base[selected]).sum())
                self.assertAlmostEqual(view["baseline_change"], (b-old_b)/denominator, places=13)
                for index in range(5):
                    p = float(np.square(actual-reduced["predictions"][index]).sum())
                    f = float(np.square(actual-full_pred[index, selected]).sum())
                    direct = (f-p)/denominator
                    own_delta_difference = (view["bank"]["lambda_rows"][index]["delta_R2"]
                        - view["reference_own_baseline_bank"]["lambda_rows"][index]["delta_R2"])
                    self.assertAlmostEqual(own_delta_difference, direct+view["baseline_change"], places=13)
                    reference_row = view["reference_bank"]["lambda_rows"][index]
                    self.assertAlmostEqual(reference_row["delta_R2"], (b-f)/denominator, places=13)
                    self.assertAlmostEqual(reference_row["wrong_pair_SSE"],
                        wrong_pair_sse(actual, full_pred[index, selected], labels[selected]), places=9)
        for corruption in ("target", "direction", "identity", "nonfinite"):
            broken = copy.deepcopy(reference)
            if corruption == "target":
                broken["actual_xy"][0, 0, 0] += 1e-12
            elif corruption == "direction":
                broken["directions"][0] = (broken["directions"][0]+1)%8
            elif corruption == "identity":
                broken["trial_ids"][0] = broken["trial_ids"][1]
            else:
                broken["candidate_xy_by_lambda"][0, 0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                matched.align_reference(broken, y, labels, ids)

    def test_outer_exclusion_direct_contrasts_and_ordered_duplicate_charges(self):
        def raw_bank(recipe, denominator, baseline, loss_at):
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "denominator": denominator,
                "baseline_SSE": baseline*denominator, "baseline_R2": 1-baseline,
                "lambda_rows": [{"lambda": penalty, "candidate_SSE": loss_at(i)*denominator,
                    "wrong_pair_SSE": (loss_at(i)+.02)*denominator,
                    "candidate_R2": 1-loss_at(i), "delta_R2": baseline-loss_at(i),
                    "identity_increment": .02} for i, penalty in enumerate(LAMBDAS)]}
        sessions = {}
        for session in SESSIONS:
            sessions[session] = {}
            all_ids = [f"{session}_trial{i+1:06d}" for i in range(32)]
            for index, recipe in enumerate(matched.RECIPES):
                views = []
                for view in range(8):
                    denominator = 16.*(view+1)
                    support = {"trial_ids": all_ids if index == 0 else all_ids[:24],
                        "electrode_ids": [1, 2, 3] if index == 1 else [1, 2], "full_support": False}
                    views.append({"view": view, "support": support, "fit_key": f"fit{index}",
                        "bank": raw_bank(recipe, denominator, 2., lambda i: 2.-.01*i),
                        "reference_bank": raw_bank(matched.REFERENCE, denominator, 2., lambda i: 1.9-.005*i),
                        "reference_own_baseline_bank": raw_bank(matched.REFERENCE, denominator, 1.5, lambda i: 1.9-.005*i),
                        "baseline_change": .5, "fit_diagnostics": [{"held_direction": d,
                            "fitting_bin_rows": 300, "trace": 8., "numerical_rank": 8,
                            "ridge_df_by_lambda": [{"lambda": p, "effective_df": 8.-i}
                                                   for i, p in enumerate(LAMBDAS)]} for d in range(8)]})
                sessions[session][recipe] = matched.average_views(views, recipe)
        reference = {"recipe": matched.REFERENCE, "sessions": [{"session_id": s,
            "lambda_index": 0, "lambda": LAMBDAS[0],
            "outer_training_sessions": [other for other in SESSIONS if other != s]} for s in SESSIONS]}
        held = SESSIONS[0]
        for recipe in matched.RECIPES:
            for index, row in enumerate(sessions[held][recipe]["lambda_rows"]):
                row["delta_R2"] = 1e6*(5-index)
        result = matched.summarize_banks(sessions, reference)
        self.assertEqual(result["scientific_trial_count_by_recipe"], dict.fromkeys(matched.RECIPES, 1))
        self.assertEqual(result["total_valid_primary_trials_including_prior"], 30)
        self.assertEqual(result["post_initial_alternative_checks"], 14)
        self.assertEqual(result["post_initial_null_controls"], 11)
        self.assertEqual(result["null_control_trial_count_increment"], 0)
        self.assertEqual([row["completed_count"] for row in result["prospective_patience_transitions"]], [28, 29, 30])
        self.assertEqual([row["patience"] for row in result["prospective_patience_transitions"]], ["not_started", 1, 2])
        self.assertFalse(result["incumbent_reset"])
        self.assertFalse(result["promotion_authorized"])
        for panel in result["panels"]:
            row = next(r for r in panel["sessions"] if r["session_id"] == held)
            self.assertEqual(row["lambda_index"], 4)
            self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
            self.assertTrue(all(view["lambda"] == LAMBDAS[4] for view in row["selected_view_scores"]))
            contrasts = result["reference_contrasts"][panel["recipe"]]
            independent = next(r for r in contrasts["independently_selected"]["sessions"] if r["session_id"] == held)
            matched_row = next(r for r in contrasts["diagnostic_lambda_matched"]["sessions"] if r["session_id"] == held)
            self.assertAlmostEqual(independent["delta_R2"], -.06, places=13)
            self.assertAlmostEqual(matched_row["delta_R2"], -.08, places=13)
            self.assertEqual(independent["full_HW_lambda"], LAMBDAS[0])
            self.assertEqual(matched_row["full_HW_lambda"], LAMBDAS[4])
        # If the channel axis is globally unchanged, C is the old full recipe
        # and CT is exactly T; one new recipe must not turn into three charges.
        same = copy.deepcopy(sessions)
        for banks in same.values():
            for view in banks[matched.RECIPES[0]]["views"]:
                view["support"]["full_support"] = True
            for t, ct in zip(banks[matched.RECIPES[1]]["views"], banks[matched.RECIPES[2]]["views"]):
                t["support"]["electrode_ids"] = [1, 2]
                ct["support"] = copy.deepcopy(t["support"])
        charges, duplicates = matched.recipe_charges(same)
        self.assertEqual(list(charges.values()), [0, 1, 0])
        self.assertEqual(duplicates[matched.RECIPES[0]], "full_HW")
        self.assertEqual(duplicates[matched.RECIPES[2]], matched.RECIPES[1])
        collapsed = matched.summarize_banks(same, reference)
        self.assertEqual(collapsed["total_valid_primary_trials_including_prior"], 28)
        self.assertTrue(all(row["patience"] == "not_started" for row in collapsed["prospective_patience_transitions"]))
        # Literal ties retain the original stronger-penalty rule.
        for banks in sessions.values():
            for bank in banks.values():
                for row in bank["lambda_rows"]:
                    row["delta_R2"] = .01
        tied = matched.summarize_banks(sessions, reference)
        self.assertTrue(all(row["lambda_index"] == 4 for panel in tied["panels"] for row in panel["sessions"]))


if __name__ == "__main__":
    unittest.main()
