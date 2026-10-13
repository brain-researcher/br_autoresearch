"""Independent nine-band support, full-Gram control and accounting fixtures."""
import copy
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

import all_band_reader as reader
import multichannel_reader as old_reader
import primary_all_band as all_band
import primary_lmp_high as joint
from method_components import DirectionTime
from primary_multichannel import guide_columns
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(seed=155001):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 3)
    combined = np.stack([rng.normal(size=(18, 15)).T.astype(np.float64) for _ in labels])
    blocks = reader.split_blocks(combined, 2)
    blocks["added"][:, :, 0] = blocks["lmp"][:, :, 0]+.1*blocks["added"][:, :, 0]
    y = rng.normal(size=(24, 15, 2))+blocks["lmp"]+.4*blocks["added"][:, :, :2]
    y[:, :, 0] += 4*np.cos(labels*np.pi/4)[:, None]
    ids = np.array([f"example_trial{i+1:06d}" for i in range(24)])
    return blocks, y, labels, ids, [2, 7]


class AllBandTest(unittest.TestCase):
    def test_fixed_nine_band_layout_exact_GUIDE_identity_and_missing_support_refusal(self):
        bands = ((0., 0.), (100., 200.), (200., 400.), (.5, 4.), (4., 8.),
                 (8., 12.), (12., 25.), (25., 50.), (50., 100.))
        self.assertEqual(tuple(all_band.ALL_BANDS), bands)
        canonical = np.array([[e, *band] for e in (2, 7) for band in bands])
        rearrangement = np.random.default_rng(155002).permutation(18)
        guide = canonical[rearrangement]
        selected = guide_columns(guide, [2, 7], bands)
        np.testing.assert_array_equal(guide[selected], canonical)
        combined = np.broadcast_to(np.arange(18), (3, 15, 18)).copy()
        blocks = reader.split_blocks(combined, 2)
        for name, expected in (("lmp", [0, 9]), ("high", [1, 2, 10, 11]),
                               ("added", [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17])):
            np.testing.assert_array_equal(blocks[name], combined[:, :, expected])
        for damaged in (guide[:-1], np.vstack([guide, guide[0]])):
            with self.assertRaises(ValueError): guide_columns(damaged, [2, 7], bands)

    def test_reader_qualifies_all_trials_before_values_and_replays_old_J_raw_layout(self):
        session = "Chewie_20150313"
        channels = [2, 7]
        rows = [{"trial_id": f"{session}_trial{i+1:06d}", "source_trial_index_zero_based": i,
                 "direction": i, "idx_movement_on": 3} for i in range(2)]
        previous = {"recipe": reader.SCALAR_RECIPE, "status": "complete_session_score_bank",
                    "session_id": session, "source": {"complete_trials": 2, "common_LMP_electrodes": channels}}
        canonical = np.asarray([[e, *band] for e in channels for band in all_band.ALL_BANDS]+[[2, 400., 600.]])
        rng = np.random.default_rng(155008)
        guides = [canonical[rng.permutation(len(canonical))] for _ in range(2)]
        canonical_values = rng.normal(size=(2, 19, 21))*np.geomspace(1e-6, 1e6, 21)
        targets = rng.normal(size=(2, 2, 21))
        case = self
        for dtype in (np.float32, np.float64):
            arrays = []
            for trial, guide in enumerate(guides):
                indices = [next(i for i, row in enumerate(canonical) if np.array_equal(row, actual)) for actual in guide]
                arrays.append(canonical_values[trial, indices].astype(dtype))
            def run(module, damaged=False):
                events, slices = [], []
                current_guides = [g.copy() for g in guides]
                if damaged:
                    missing = next(i for i, row in enumerate(current_guides[1]) if np.array_equal(row, [7, 25, 50]))
                    current_guides[1][missing, 1:] = [26, 50]
                class Node:
                    def __init__(self, values, field, trial):
                        self.values, self.field, self.trial = values, field, trial
                        self.shape, self.ndim = values.shape, values.ndim
                    def __getitem__(self, key):
                        slices.append((self.field, self.trial, key))
                        return self.values[key]
                class Session:
                    def __init__(self, *args): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def metadata(self, trial):
                        return {**rows[trial], "direction_circular_error": 0., "result": "R"}
                    def field(self, trial, field):
                        if field == "M1_lfp_guide":
                            events.append(("guide", trial))
                            return current_guides[trial].T
                        case.assertEqual(events[:2], [("guide", 0), ("guide", 1)])
                        case.assertIn(field, ("M1_lfp", "vel"))
                        events.append((field, trial))
                        return Node(arrays[trial] if field == "M1_lfp" else targets[trial], field, trial)
                with patch.object(module, "M1Session", Session), \
                        patch.object(module, "trial_records", return_value=rows), \
                        patch.object(Path, "read_text", return_value=json.dumps(previous)):
                    if damaged:
                        with case.assertRaises(ValueError):
                            module.load_session(Path("unopened"), Path("table"), Path("reference"), session, lambda: None)
                        case.assertEqual(events, [("guide", 0), ("guide", 1)])
                        case.assertEqual(slices, [])
                        return None
                    packet = module.load_session(Path("unopened"), Path("table"), Path("reference"), session, lambda: None)
                for field, trial, key in slices:
                    case.assertEqual(key[1], slice(3, 18))
                    if field == "M1_lfp":
                        case.assertTrue(np.all(np.diff(key[0]) > 0))
                        selected = current_guides[trial][key[0]]
                        case.assertFalse(any(np.array_equal(row[1:], [400, 600]) for row in selected))
                return packet
            new = run(reader)
            old = run(old_reader)
            for name, old_name in (("lmp", "lmp_full_channel_ridge_v1"), ("high", "high_full_channel_ridge_v1")):
                np.testing.assert_array_equal(new[0][name], old[0][old_name])
                self.assertEqual(new[0][name].strides, old[0][old_name].strides)
                for train in (np.array([True, False]), np.array([False, True])):
                    np.testing.assert_array_equal(new[0][name][train].mean(axis=(0, 1)), old[0][old_name][train].mean(axis=(0, 1)))
                    np.testing.assert_array_equal(new[0][name][train].std(axis=(0, 1)), old[0][old_name][train].std(axis=(0, 1)))
            np.testing.assert_array_equal(new[1], targets[:, :, 3:18].transpose(0, 2, 1))
            self.assertEqual(new[4]["selected_logical_source_bytes"], 2*(18*15*np.dtype(dtype).itemsize+2*15*8))
            self.assertFalse(new[4]["source_spike_values_read"])
            run(reader, damaged=True)

    def test_fitting_only_normalizers_preserve_J_paths_and_algebraic_prefix_embedding(self):
        blocks, y, labels, ids, channels = example(155003)
        train, test = labels != 4, labels == 4
        standardized, states = all_band.prepare_split(blocks["lmp"], blocks["high"], blocks["added"], train)
        original_lmp, original_high = joint.prepare_split(blocks["lmp"], blocks["high"], train)
        np.testing.assert_array_equal(standardized["lmp"], original_lmp)
        np.testing.assert_array_equal(standardized["high"], original_high)
        changed = {name: value.copy(order="K") for name, value in blocks.items()}
        for values in changed.values(): values[test] += 1e9
        other, other_states = all_band.prepare_split(changed["lmp"], changed["high"], changed["added"], train)
        for name, values in blocks.items():
            mean, sd = values[train].mean(axis=(0, 1)), values[train].std(axis=(0, 1))
            np.testing.assert_array_equal(states[name]["physical_mean"], mean)
            np.testing.assert_array_equal(states[name]["physical_sd"], sd)
            np.testing.assert_array_equal(standardized[name], (values-mean)/np.where(sd == 0, 1., sd))
            np.testing.assert_array_equal(standardized[name][train], other[name][train])
            for key in ("physical_mean", "physical_sd"):
                np.testing.assert_array_equal(states[name][key], other_states[name][key])
        fitting, evaluation, center = all_band.joint_arm(standardized, train, test)
        original = joint.joint_arm(original_lmp, original_high, train)
        np.testing.assert_allclose(fitting[:, :, :6], original[train], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(evaluation[:, :, :6], original[test], rtol=2e-14, atol=2e-14)
        coefficient = np.random.default_rng(155004).normal(size=(6, 2))
        embedded = np.vstack([coefficient, np.zeros((12, 2))])
        np.testing.assert_allclose(evaluation@embedded, original[test]@coefficient, rtol=2e-13, atol=2e-14)

    def test_native_canonical_maps_use_one_ordinary_whole_trajectory_draw_per_direction(self):
        blocks, y, labels, ids, _ = example(155005)
        held, train = 6, labels != 6
        ids, directions = ids[train], labels[train]
        mapping, records = all_band.trial_map(ids, directions, "example", held, 3)
        np.testing.assert_array_equal(directions[mapping], directions)
        self.assertEqual(sorted(mapping.tolist()), list(range(len(ids))))
        for row in records:
            literal = f"ep05-lmp-all-band-added-training-label-v1|20261004|example|6|3|{row['direction']}"
            expected = row["recipient_trial_ids"].copy()
            rng = random.Random()
            rng.seed(literal, version=2)
            rng.shuffle(expected)
            self.assertEqual(row["seed_literal"], literal)
            self.assertEqual(row["donor_trial_ids"], expected)
        reverse = np.arange(len(ids))[::-1]
        reversed_map, reversed_records = all_band.trial_map(ids[reverse], directions[reverse], "example", held, 3)
        self.assertEqual(records, reversed_records)
        np.testing.assert_array_equal(ids[reverse][reversed_map], ids[mapping][reverse])
        class Identity:
            def seed(self, literal, version): pass
            def shuffle(self, values): pass
        with patch.object(all_band.random, "Random", Identity):
            identity, _ = all_band.trial_map(ids, directions, "example", held, 3)
        np.testing.assert_array_equal(identity, np.arange(len(ids)))
        malformed = ids.copy()
        malformed[0] = "wrong_trial000001"
        with self.assertRaises(ValueError): all_band.trial_map(malformed, directions, "example", held, 3)

    def test_only_fitting_added_block_moves_and_complete_own_Gram_solves_match_OLS(self):
        blocks, y, labels, ids, _ = example(155006)
        train, test = labels != 2, labels == 2
        standardized, _ = all_band.prepare_split(blocks["lmp"], blocks["high"], blocks["added"], train)
        frozen = {name: value.copy() for name, value in standardized.items()}
        mapping, _ = all_band.trial_map(ids[train], labels[train], "example", 2, 0)
        paired, paired_eval, _ = all_band.joint_arm(standardized, train, test)
        fitting, evaluation, center = all_band.joint_arm(standardized, train, test, mapping)
        raw_fit = np.concatenate([standardized["lmp"][train], standardized["high"][train],
                                  standardized["added"][train][mapping]], axis=-1)
        raw_eval = np.concatenate([standardized[name][test] for name in ("lmp", "high", "added")], axis=-1)
        expected_center = raw_fit.mean(axis=(0, 1))
        np.testing.assert_array_equal(center, expected_center)
        np.testing.assert_array_equal(fitting, raw_fit-expected_center)
        np.testing.assert_array_equal(evaluation, raw_eval-expected_center)
        for name in frozen: np.testing.assert_array_equal(standardized[name], frozen[name])
        z, z0 = fitting.reshape(-1, 18), paired.reshape(-1, 18)
        np.testing.assert_allclose(z[:, :6].T@z[:, :6], z0[:, :6].T@z0[:, :6], rtol=2e-14, atol=2e-12)
        np.testing.assert_allclose(z[:, 6:].T@z[:, 6:], z0[:, 6:].T@z0[:, 6:], rtol=2e-14, atol=2e-12)
        self.assertGreater(np.linalg.norm(z[:, :6].T@z[:, 6:]-z0[:, :6].T@z0[:, 6:]), 1.)
        residual = np.random.default_rng(155007).normal(size=(len(z), 2))
        predicted = all_band.ridge_predictions(fitting, evaluation, residual)
        for index, penalty in enumerate(LAMBDAS):
            design = np.vstack([z, np.sqrt(len(z)*penalty)*np.eye(18)])
            response = np.vstack([residual, np.zeros((18, 2))])
            beta = np.linalg.lstsq(design, response, rcond=None)[0]
            np.testing.assert_allclose(predicted[index], evaluation.reshape(-1, 18)@beta,
                                       rtol=2e-11, atol=2e-12)

    def test_all72_heads_keep_true_native_U_and_control_scores_average_eight_losses(self):
        blocks, y, labels, ids, channels = example(155009)
        observed = []
        original = all_band.ridge_predictions
        def head(fitting, evaluation, residual, check):
            predicted = original(fitting, evaluation, residual, check)
            observed.append((fitting.copy(), evaluation.copy(), residual.copy(), predicted.copy()))
            return predicted
        with patch.object(all_band, "ridge_predictions", side_effect=head):
            banks, baseline, predictions = all_band.fit_banks(blocks["lmp"], blocks["high"], blocks["added"],
                y, labels, ids, "example", electrode_ids=channels)
        self.assertEqual(len(observed), 72)
        different_full_gram = False
        for held in range(8):
            train, test = labels != held, labels == held
            template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
            residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
            intercept = residual.mean(axis=0)
            true_u = residual-intercept
            base = template.predict(labels[test]*np.pi/4)+intercept
            np.testing.assert_array_equal(baseline[test], base)
            paired_fit = observed[9*held][0].reshape(-1, 18)
            for arm in range(9):
                fitting, evaluation, target, head_prediction = observed[9*held+arm]
                np.testing.assert_array_equal(target, true_u)
                expected = np.asarray([base+prediction.reshape(base.shape) for prediction in head_prediction])
                actual = predictions[all_band.RECIPES[0]][:, test] if arm == 0 else predictions[all_band.RECIPES[1]][arm-1][:, test]
                np.testing.assert_array_equal(actual, expected)
                if arm:
                    phi = fitting.reshape(-1, 18)
                    different_full_gram |= not np.array_equal(phi.T@phi, paired_fit.T@paired_fit)
            row = banks[all_band.RECIPES[0]]["direction_fit_metadata"][held]
            self.assertEqual(len(row["control_covariances"]), 8)
            self.assertEqual(len(row["training_added_maps"]), 8)
        self.assertTrue(different_full_gram)
        control = predictions[all_band.RECIPES[1]]
        for index in range(len(LAMBDAS)):
            row = banks[all_band.RECIPES[1]]["lambda_rows"][index]
            expected_loss = np.mean(np.sum((control[:, index]-y)**2, axis=(1, 2, 3)))
            expected_wrong = np.mean([wrong_pair_sse(y, prediction, labels) for prediction in control[:, index]])
            self.assertAlmostEqual(row["candidate_SSE"], expected_loss, places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], expected_wrong, places=9)
        mean_prediction_loss = np.sum((control[:, 0].mean(axis=0)-y)**2)
        self.assertGreater(banks[all_band.RECIPES[1]]["lambda_rows"][0]["candidate_SSE"]-mean_prediction_loss, 1e-8)

    def test_zero_added_energy_embeds_J_and_identity_maps_do_not_create_new_control(self):
        blocks, y, labels, ids, channels = example(155010)
        zero = np.zeros_like(blocks["added"])
        banks, baseline, predictions = all_band.fit_banks(blocks["lmp"], blocks["high"], zero,
            y, labels, ids, "example", electrode_ids=channels)
        old, old_base, old_predictions = joint.fit_banks(blocks["lmp"], blocks["high"], y, labels, ids, "example")
        np.testing.assert_array_equal(baseline, old_base)
        self.assertTrue(banks[all_band.RECIPES[0]]["structural_J_duplicate"])
        self.assertTrue(banks[all_band.RECIPES[1]]["structural_J_duplicate"])
        self.assertTrue(banks[all_band.RECIPES[0]]["all_controls_equal_candidate"])
        np.testing.assert_allclose(predictions[all_band.RECIPES[0]], old_predictions[joint.RECIPES[0]], rtol=2e-11, atol=2e-12)
        class Identity:
            def seed(self, literal, version): pass
            def shuffle(self, values): pass
        with patch.object(all_band.random, "Random", Identity):
            same, _, same_predictions = all_band.fit_banks(blocks["lmp"], blocks["high"], blocks["added"],
                y, labels, ids, "example", electrode_ids=channels)
        self.assertFalse(same[all_band.RECIPES[0]]["structural_J_duplicate"])
        self.assertTrue(same[all_band.RECIPES[0]]["all_controls_equal_candidate"])
        self.assertTrue(same[all_band.RECIPES[1]]["structural_paired_duplicate"])
        for prediction in same_predictions[all_band.RECIPES[1]]:
            np.testing.assert_array_equal(prediction, same_predictions[all_band.RECIPES[0]])

    def test_shared_outer_lambda_four_anchor_contrasts_and_zero_one_two_charges(self):
        def bank(recipe, paired):
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8,
                "structural_J_duplicate": False, "all_controls_equal_candidate": False,
                "structural_paired_duplicate": False,
                "lambda_rows": [{"lambda": penalty, "delta_R2": .01 if paired else .04-.01*i,
                    "identity_increment": .01, "candidate_SSE": 5., "wrong_pair_SSE": 5.1,
                    "candidate_R2": .5} for i, penalty in enumerate(LAMBDAS)]}
        sessions = {s: {all_band.RECIPES[0]: bank(all_band.RECIPES[0], True),
                    all_band.RECIPES[1]: all_band.average_control_banks([bank(all_band.RECIPES[1], False) for _ in range(8)])}
                    for s in SESSIONS}
        anchors = {name: {s: bank(recipe, False) for s in SESSIONS} for name, recipe in all_band.ANCHORS.items()}
        anchor_panels = {name: {"recipe": recipe, "sessions": [
            {"session_id": s, "animal": ANIMALS[s], **anchors[name][s]["lambda_rows"][0]} for s in SESSIONS]}
            for name, recipe in all_band.ANCHORS.items()}
        held = SESSIONS[0]
        for i, row in enumerate(sessions[held][all_band.RECIPES[0]]["lambda_rows"]):
            row["delta_R2"] = 1e7*(5-i)
        for duplicate, same, candidate_charge, control_charge in ((False, False, 1, 1),
                (False, True, 1, 0), (True, True, 0, 0)):
            for banks in sessions.values():
                banks[all_band.RECIPES[0]].update(structural_J_duplicate=duplicate, all_controls_equal_candidate=same)
                banks[all_band.RECIPES[1]].update(structural_J_duplicate=duplicate, structural_paired_duplicate=same)
            result = all_band.summarize_banks(sessions, anchors, anchor_panels)
            self.assertEqual(result["scientific_trial_count_increment"], candidate_charge+control_charge)
            self.assertEqual(result["control_trial_count_increment"], control_charge)
            self.assertEqual(result["scientific_trial_count_by_recipe"],
                             dict(zip(all_band.RECIPES, (candidate_charge, control_charge))))
            self.assertEqual(set(result["anchor_contrasts"]), {
                "joint_independently_selected", "joint_all_band_lambda_matched",
                "lmp_independently_selected", "lmp_all_band_lambda_matched"})
            for panel in result["panels"]:
                self.assertFalse(panel["promotion_authorized"])
                self.assertFalse(panel["search_complete"])
                row = next(row for row in panel["sessions"] if row["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
            for name in all_band.ANCHORS:
                matched = next(row for row in result["matched_lambda_anchor_sessions"][name] if row["session_id"] == held)
                original = next(row for row in anchor_panels[name]["sessions"] if row["session_id"] == held)
                self.assertEqual(matched["lambda"], LAMBDAS[4])
                self.assertEqual(original["lambda"], LAMBDAS[0])


if __name__ == "__main__":
    unittest.main()
