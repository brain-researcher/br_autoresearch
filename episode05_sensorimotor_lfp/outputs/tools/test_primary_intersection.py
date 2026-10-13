"""Independent fixed-support, native-layout, reuse and I/R accounting fixtures."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import primary_intersection as intersection
import primary_matched_budget as matched
import qualify_primary_intersection as qualifier
import run_primary_intersection as runner
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from test_primary_history2_repeatability_weight import data


def description(session=SESSIONS[0], *, full=False):
    x, y, labels, _, electrodes = data(215001)
    ids = np.array([f"{session}_trial{i+1:06d}" for i in range(len(y))])
    selected = electrodes if full else [electrodes[0], electrodes[-1]]
    guide = [[e, 0., 0.] for e in electrodes]
    units = [[e, 0.] for e in selected]+[[selected[0], 3.], [99, -1.]]
    joined = qualifier.guide_intersection(guide, units, electrodes)
    rows = [{"trial_id": value, "direction": int(direction), "source_trial_index_zero_based": i}
            for i, (value, direction) in enumerate(zip(ids, labels))]
    supports = qualifier.support_sets(electrodes, selected, session)
    unique = len({tuple(row["electrode_ids"]) for row in supports if not row["full_support"]})
    width = 4*len(selected)
    result = {**joined, "session_id": session, "animal": ANIMALS[session], "complete_trials": len(y),
        "trial_ids": ids.tolist(), "trial_directions": labels.tolist(),
        "source_trial_indices_zero_based": list(range(len(y))), "direction_counts": [3]*8,
        "lfp_guide": guide, "constant_guides_on_all_original_trials": True,
        "spike_values_read": False, "LFP_XY_values_read": False, "scores_computed": False,
        "support_sets": supports, "unique_new_fit_supports": unique,
        "workload_proxy": unique*(105*len(y)*width**2+8*width**3)}
    return result, rows, (x, y, labels, ids)


def reference_packet(x, y, labels, ids):
    baseline, predictions, metadata = matched.fit_view(x, y, labels)
    # NPZ storage yields C-order saved arrays; exact old scoring was produced
    # from the original time-fast y/B layouts.
    return {"actual_xy": np.array(y, order="C"), "baseline_xy": np.array(baseline, order="C"),
        "directions": labels.copy(), "trial_ids": ids.copy(), "candidate_xy_by_lambda": predictions,
        "fold_metadata": metadata, "score_bank": score_bank(y, labels, baseline, predictions, intersection.REFERENCE)}


def complete_panel():
    desc, _, (x, y, labels, ids) = description()
    fits, routes = intersection.fit_supports(x, y, labels, ids, desc)
    reference = reference_packet(x, y, labels, ids)
    banks, _ = intersection.finalize_banks(y, labels, ids, desc, fits, routes, reference)
    sessions = {session: copy.deepcopy(banks) for session in SESSIONS}
    rows = []
    for session in SESSIONS:
        old = banks[intersection.RECIPES[0]]["full_HW_bank"]
        index = max(range(5), key=lambda i: (old["lambda_rows"][i]["delta_R2"], LAMBDAS[i]))
        rows.append({"session_id": session, "animal": ANIMALS[session], "lambda_index": index,
            "outer_training_sessions": [s for s in SESSIONS if s != session], **old["lambda_rows"][index]})
    return sessions, {"recipe": intersection.REFERENCE, "sessions": rows}


def set_scores(bank, increments):
    for row, increment in zip(bank["lambda_rows"], increments):
        correct = bank["baseline_SSE"]-increment*bank["denominator"]
        row.update(delta_R2=increment, candidate_SSE=correct, candidate_R2=1-correct/bank["denominator"],
                   wrong_pair_SSE=correct+.001*bank["denominator"], identity_increment=.001)


def summarize(sessions, reference, *, valid=33, patience=0):
    return intersection.summarize_banks(sessions, reference, prior_valid=valid,
        prior_patience=patience, prior_alternatives=14, prior_nulls=12)


class IntersectionTest(unittest.TestCase):
    def test_all_sixteen_metadata_original_order_queues_and_nonempty_support_required(self):
        rows, records = [], {}
        for session in SESSIONS:
            row, table, _ = description(session)
            rows.append(row)
            records[session] = table
        metadata = {"status": "complete_primary_intersection_metadata", "sessions": rows,
            "all_sixteen_qualified_before_values": True, "scientific_trial_count_increment": 0,
            "spike_values_read": False, "LFP_XY_values_read": False, "scores_computed": False,
            "pilot_session": qualifier.choose_pilot(rows), "global_full_HW_duplicate": False}
        result = intersection.validate_metadata(metadata, records)
        self.assertEqual(list(result), list(SESSIONS))
        self.assertEqual(result[SESSIONS[0]]["intersection_electrode_ids"], [2, 11])
        self.assertIn([2, 0.], result[SESSIONS[0]]["unit_guide_pairs"])
        for damage in ("partial", "order", "queue", "trial", "empty", "numerical_read"):
            changed = copy.deepcopy(metadata)
            if damage == "partial":
                changed["sessions"].pop()
            elif damage == "order":
                changed["sessions"].reverse()
            elif damage == "queue":
                changed["sessions"][0]["support_sets"][1]["electrode_queue"].reverse()
            elif damage == "trial":
                changed["sessions"][0]["trial_ids"].reverse()
            elif damage == "empty":
                changed["sessions"][0]["unit_guide_pairs"] = [[99., 0.]]
            else:
                changed["LFP_XY_values_read"] = True
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                intersection.validate_metadata(changed, records)

    def test_channel_only_fit_preserves_original_y_labels_and_reuses_exact_supports(self):
        desc, _, (x, y, labels, ids) = description()
        seen = []
        def fake_fit(selected, actual, directions, check):
            self.assertIs(actual, y)
            self.assertIs(directions, labels)
            expected_supports = {tuple(row["electrode_positions"]) for row in desc["support_sets"]}
            matching = [positions for positions in expected_supports if np.array_equal(selected, x[:, :, positions])]
            self.assertEqual(len(matching), 1)
            self.assertEqual(selected.strides, x[:, :, matching[0]].strides)
            seen.append(matching[0])
            return np.zeros_like(y), np.zeros((5, *y.shape)), []
        with patch.object(intersection, "fit_view", side_effect=fake_fit), \
                patch.object(matched, "subset_arrays", side_effect=AssertionError("Trial/Y copier is forbidden")):
            fits, routes = intersection.fit_supports(x, y, labels, ids, desc)
        self.assertEqual(len(routes), 9)
        self.assertEqual(len(fits), desc["unique_new_fit_supports"])
        self.assertEqual(len(seen), len(set(seen)))
        for i, left in enumerate(desc["support_sets"]):
            for j, right in enumerate(desc["support_sets"]):
                self.assertEqual(routes[i] == routes[j], left["electrode_ids"] == right["electrode_ids"])
        self.assertIs(intersection.fit_view, matched.fit_view)

    def test_full_support_reuses_original_predictions_and_native_layout_without_any_fit(self):
        desc, _, (x, y, labels, ids) = description(full=True)
        reference = reference_packet(x, y, labels, ids)
        with patch.object(intersection, "fit_view", side_effect=AssertionError("Full HW must not be refit")):
            fits, routes = intersection.fit_supports(x, y, labels, ids, desc)
            banks, payload = intersection.finalize_banks(y, labels, ids, desc, fits, routes, reference)
        self.assertEqual(fits, {})
        self.assertEqual(routes, ["full_HW"]*9)
        self.assertEqual(payload["baseline_xy"].strides, y.strides)
        np.testing.assert_array_equal(payload["baseline_xy"], reference["baseline_xy"])
        for recipe in intersection.RECIPES:
            bank = banks[recipe]
            self.assertEqual(bank["baseline_SSE"], reference["score_bank"]["baseline_SSE"])
            self.assertEqual(bank["denominator"], reference["score_bank"]["denominator"])
            for view in bank["views"]:
                self.assertEqual(view["bank"]["lambda_rows"], reference["score_bank"]["lambda_rows"])
        sessions = {s: copy.deepcopy(banks) for s in SESSIONS}
        counts, duplicate = intersection.recipe_charges(sessions)
        self.assertEqual(counts, dict.fromkeys(intersection.RECIPES, 0))
        self.assertEqual(duplicate, dict.fromkeys(intersection.RECIPES, "full_HW"))

    def test_reduced_fits_preserve_exact_B_Y_denominator_and_average_losses_not_predictions(self):
        desc, _, (x, y, labels, ids) = description()
        reference = reference_packet(x, y, labels, ids)
        fits, routes = intersection.fit_supports(x, y, labels, ids, desc)
        banks, payload = intersection.finalize_banks(y, labels, ids, desc, fits, routes, reference)
        self.assertEqual(payload["baseline_xy"].strides, y.strides)
        candidate, control = (banks[recipe] for recipe in intersection.RECIPES)
        for bank in (candidate, control):
            for key in ("baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
                self.assertEqual(bank[key], reference["score_bank"][key])
        for index, row in enumerate(control["lambda_rows"]):
            for key in ("candidate_SSE", "wrong_pair_SSE"):
                expected = float(np.mean([view["bank"]["lambda_rows"][index][key] for view in control["views"]]))
                self.assertEqual(row[key], expected)
        self.assertEqual(len(control["support_seed_rows"]), 8)
        averaged_predictions = np.mean([fits[key]["predictions"] for key in routes[1:]], axis=0)
        ensemble = score_bank(y, labels, payload["baseline_xy"], averaged_predictions, intersection.RECIPES[1])
        self.assertGreater(control["lambda_rows"][0]["candidate_SSE"]-ensemble["lambda_rows"][0]["candidate_SSE"], .01)
        changed = copy.deepcopy(fits)
        changed[routes[0]]["baseline"][0, 0, 0] += 1.
        with self.assertRaises(ValueError):
            intersection.finalize_banks(y, labels, ids, desc, changed, routes, reference)
        changed_ref = copy.deepcopy(reference)
        changed_ref["trial_ids"] = changed_ref["trial_ids"][::-1]
        with self.assertRaises(ValueError):
            intersection.finalize_banks(y, labels, ids, desc, fits, routes, changed_ref)

    def test_I_outer_selection_controls_every_R_draw_and_equal_animal_stronger_ties(self):
        sessions, reference = complete_panel()
        for banks in sessions.values():
            set_scores(banks[intersection.RECIPES[0]], [.02, .01, 0., -.01, -.02])
            set_scores(banks[intersection.RECIPES[1]], [-.02, -.01, 0., .01, .02])
        held = SESSIONS[0]
        set_scores(sessions[held][intersection.RECIPES[0]], [-10., -10., -10., -10., .05])
        result = summarize(sessions, reference)
        panels = {panel["recipe"]: panel for panel in result["panels"]}
        selected = {row["session_id"]: row for row in panels[intersection.RECIPES[0]]["sessions"]}
        random = {row["session_id"]: row for row in panels[intersection.RECIPES[1]]["sessions"]}
        self.assertEqual(selected[held]["lambda_index"], 0)
        for session, row in selected.items():
            training = [s for s in SESSIONS if s != session]
            scores = [sum(float(np.mean([sessions[s][intersection.RECIPES[0]]["lambda_rows"][i]["delta_R2"]
                                         for s in training if ANIMALS[s] == animal]))
                          for animal in ("Mihili", "Chewie"))/2 for i in range(5)]
            chosen = max(range(5), key=lambda i: (scores[i], LAMBDAS[i]))
            self.assertEqual(row["lambda_index"], chosen)
            self.assertEqual(row["outer_training_sessions"], training)
            self.assertEqual(random[session]["lambda_index"], chosen)
            for seed in result["random_support_panels"]:
                seed_row = next(value for value in seed["sessions"] if value["session_id"] == session)
                self.assertEqual(seed_row["lambda"], LAMBDAS[chosen])
        for banks in sessions.values():
            set_scores(banks[intersection.RECIPES[0]], [.01]*5)
        tied = summarize(sessions, reference)
        self.assertTrue(all(row["lambda"] == 10. for row in tied["panels"][0]["sessions"]))
        for session, banks in sessions.items():
            value = .02 if ANIMALS[session] == "Mihili" else -.015
            set_scores(banks[intersection.RECIPES[0]], [value, -.1, -.1, -.1, 0.])
        balanced = summarize(sessions, reference)
        self.assertTrue(all(row["lambda"] == .001 for row in balanced["panels"][0]["sessions"]))

    def test_common_D_direct_anchors_and_ordered_zero_one_two_nonnull_counts(self):
        sessions, reference = complete_panel()
        result = summarize(sessions, reference, valid=33, patience=0)
        self.assertEqual(result["scientific_trial_count_by_recipe"], dict.fromkeys(intersection.RECIPES, 1))
        self.assertEqual(result["literal_null_control_count_increment"], 0)
        self.assertEqual(result["alternative_explanation_count_increment"], 2)
        self.assertFalse(result["incumbent_reset"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["spike_contamination_identified"])
        self.assertEqual([row["recipe"] for row in result["prospective_patience_transitions"]], list(intersection.RECIPES))
        self.assertEqual([row["completed_count"] for row in result["prospective_patience_transitions"]], [34, 35])
        self.assertEqual([row["patience"] for row in result["prospective_patience_transitions"]], [1, 2])
        candidate = {row["session_id"]: row for row in result["panels"][0]["sessions"]}
        old = {row["session_id"]: row for row in reference["sessions"]}
        for row in result["reference_contrasts"]["independently_selected"]["sessions"]:
            session = row["session_id"]
            denominator = sessions[session][intersection.RECIPES[0]]["denominator"]
            self.assertEqual(row["delta_R2"], (old[session]["candidate_SSE"]-candidate[session]["candidate_SSE"])/denominator)
        with self.assertRaises(ValueError):
            summarize(sessions, reference, valid=63)
        # Identical support sets remain repeated reference draws; they do not
        # acquire independent scientific charges from differing score noise.
        for banks in sessions.values():
            same = banks[intersection.RECIPES[0]]["views"][0]["support"]["electrode_ids"]
            for view in banks[intersection.RECIPES[1]]["views"]:
                view["support"]["electrode_ids"] = same.copy()
        one = summarize(sessions, reference)
        self.assertEqual(one["scientific_trial_count_by_recipe"], {intersection.RECIPES[0]: 1, intersection.RECIPES[1]: 0})
        self.assertEqual([row["patience"] for row in one["prospective_patience_transitions"]], [1, 1])
        for banks in sessions.values():
            banks[intersection.RECIPES[0]]["views"][0]["support"]["full_support"] = True
            for view in banks[intersection.RECIPES[1]]["views"]:
                view["support"]["full_support"] = True
        zero = summarize(sessions, reference)
        self.assertEqual(zero["scientific_trial_count_increment"], 0)
        self.assertEqual([row["patience"] for row in zero["prospective_patience_transitions"]], [0, 0])

    def test_runner_accepts_actual_reduced_and_full_packets_rejects_support_and_loss_drift(self):
        for full in (False, True):
            desc, _, (x, y, labels, ids) = description(full=full)
            fits, routes = intersection.fit_supports(x, y, labels, ids, desc)
            reference = reference_packet(x, y, labels, ids)
            banks, _ = intersection.finalize_banks(y, labels, ids, desc, fits, routes, reference)
            packet = {"status": "complete_intersection_sensitivity_session_banks", "session_id": SESSIONS[0],
                "global_metadata_qualified_before_source": True,
                "exact_original_HW_target_baseline_trial_score_replay": True,
                "reference_refitted": False, "sensitivity_only": True, "promotion_authorized": False,
                "qualified_support": desc, "banks": banks,
                "source": {"session_id": SESSIONS[0], "common_LMP_electrodes": desc["original_electrode_ids"],
                    "complete_trials": len(y), "native_LMP_offsets": [-3, 14], "native_target_offsets": [0, 14],
                    "lag_offsets": [0, -1, -2, -3], "native_bin_seconds": .03, "spike_values_read": False},
                "new_fit_metadata": {key: {field: fit[field] for field in ("metadata", "electrode_ids")}
                                     for key, fit in fits.items()}, "new_unique_fit_count": len(fits)}
            runner.validate_session(packet, SESSIONS[0])
            for damage in ("source_interval", "mean_SSE", "fit_count", "support_queue"):
                changed = copy.deepcopy(packet)
                if damage == "source_interval":
                    changed["source"]["native_target_offsets"] = [1, 15]
                elif damage == "mean_SSE":
                    changed["banks"][intersection.RECIPES[1]]["lambda_rows"][0]["candidate_SSE"] += 1.
                elif damage == "fit_count":
                    changed["new_unique_fit_count"] += 1
                else:
                    changed["qualified_support"]["support_sets"][1]["electrode_queue"].reverse()
                with self.subTest(full=full, damage=damage), self.assertRaises(ValueError):
                    runner.validate_session(changed, SESSIONS[0])

    def test_stale_original_selected_HW_score_selector_and_exclusion_stop(self):
        sessions, reference = complete_panel()
        for damage in ("score", "selector", "training"):
            changed = copy.deepcopy(reference)
            row = changed["sessions"][0]
            if damage == "score":
                row["candidate_SSE"] += 1.
            elif damage == "selector":
                index = (row["lambda_index"]+1) % 5
                row.update(sessions[SESSIONS[0]][intersection.RECIPES[0]]["full_HW_bank"]["lambda_rows"][index])
                row["lambda_index"] = index
            else:
                row["outer_training_sessions"][0] = row["session_id"]
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                summarize(sessions, changed)


if __name__ == "__main__":
    unittest.main()
