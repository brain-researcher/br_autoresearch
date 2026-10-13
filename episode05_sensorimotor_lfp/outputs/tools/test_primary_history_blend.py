"""Independent saved-prediction ensemble, nesting and identity fixtures."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import primary_history_blend as blend
import run_primary_history_blend as runner
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS


def time_fast(values):
    return np.stack([row.T.copy(order="C").T for row in np.asarray(values)])


def example(seed=508011):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 3)
    theta = labels*np.pi/4
    wave = np.linspace(.5, 2., 15)
    baseline = np.stack((7+np.cos(theta)[:, None]*wave,
                         -4+np.sin(theta)[:, None]*wave), axis=2)
    y = time_fast(baseline+rng.normal(size=baseline.shape))
    baseline = time_fast(baseline)
    w = np.stack([baseline+.15*(index+1)*(y-baseline)+rng.normal(scale=.2, size=y.shape)
                  for index in range(len(LAMBDAS))])
    hw = np.stack([baseline+.11*(index+1)*(y-baseline)+rng.normal(scale=.3, size=y.shape)
                   for index in range(len(LAMBDAS))])
    return y, baseline, labels, np.ascontiguousarray(w), np.ascontiguousarray(hw)


def direct_wrong(y, predictions, labels):
    total = 0.
    for direction in range(8):
        actual, fitted = y[labels == direction], predictions[labels == direction]
        losses = [float(np.square(actual-fitted[list(order)]).sum())
                  for order in itertools.permutations(range(len(actual)))
                  if all(index != donor for index, donor in enumerate(order))]
        total += float(np.mean(losses))
    return total


def synthetic_banks():
    y, baseline, labels, w, hw = example()
    template = blend.score_session(y, baseline, labels, w, hw, lambda: None)
    return {session: copy.deepcopy(template) for session in SESSIONS}


def complete_family(y, baseline, labels, w, hw):
    sessions = {session: blend.score_session(y, baseline, labels, w, hw, lambda: None) for session in SESSIONS}
    anchors = {name: {session: score_bank(y, labels, baseline, values, blend.COMPONENTS[name])
                     for session in SESSIONS} for name, values in (("W", w), ("HW", hw))}
    panels = {}
    for name in blend.COMPONENTS:
        selected = []
        for held in SESSIONS:
            training = [session for session in SESSIONS if session != held]
            scores = []
            for index in range(len(LAMBDAS)):
                animal_means = [np.mean([anchors[name][s]["lambda_rows"][index]["delta_R2"]
                                       for s in training if ANIMALS[s] == animal])
                                for animal in ("Mihili", "Chewie")]
                scores.append(float(sum(animal_means)/2))
            index = max(range(5), key=lambda i: (scores[i], LAMBDAS[i]))
            selected.append({"session_id": held, "animal": ANIMALS[held], "lambda_index": index,
                "outer_training_sessions": training, **anchors[name][held]["lambda_rows"][index]})
        panels[name] = {"recipe": blend.COMPONENTS[name], "sessions": selected}
    return sessions, anchors, panels


def saved_components(directory, values=None):
    """Small original-producer packet with deliberate C-order NPZ targets."""
    y, baseline, labels, w, hw = example(508017) if values is None else values
    session = SESSIONS[0]
    ids = np.array([f"{session}_trial{i+1:06d}" for i in range(len(y))])
    roots = {name: Path(directory)/name for name in blend.COMPONENTS}
    for name, predictions in (("W", w), ("HW", hw)):
        folder = roots[name]/session
        folder.mkdir(parents=True, exist_ok=True)
        recipe = blend.COMPONENTS[name]
        bank = score_bank(y, labels, baseline, predictions, recipe)
        bank.update(physical_electrode_ids=[1, 3, 7], physical_electrode_count=3,
                    native_XY_scaled=False)
        source = {"session_id": session, "native_bin_seconds": .03,
                  "complete_trials": len(y), "common_LMP_electrodes": [1, 3, 7]}
        if name == "W":
            source["native_offsets"] = [0, 14]
        else:
            source.update(native_LMP_offsets=[-3, 14], native_target_offsets=[0, 14],
                          lag_offsets=[0, -1, -2, -3])
        item = {"status": runner.COMPLETE_STATUSES[name], "session_id": session,
                "source": source, "banks": {recipe: bank},
                "exact_original_LMP_baseline_target_trial_replay": True,
                "exact_original_current_weight_state_and_maps_replay": True,
                "distinct_history_input_and_primary_target_windows_verified": True}
        (folder/"score_banks.json").write_text(json.dumps(item))
        np.savez(folder/(recipe+"_predictions.npz"),
                 actual_xy=np.ascontiguousarray(y), baseline_xy=np.ascontiguousarray(baseline),
                 directions=labels, trial_ids=ids, candidate_xy_by_lambda=predictions)
    return roots, ids


class HistoryBlendTest(unittest.TestCase):
    def test_endpoint_arrays_are_literal_objects_and_interior_is_full_prediction_mixture(self):
        y, baseline, _, w, hw = example()
        left, right = w[0], hw[0]
        left[0, 0] = [-0., 0.]
        right[0, 0] = [1e150, -1e150]
        self.assertIs(blend.blend_prediction(left, right, 0), left)
        self.assertIs(blend.blend_prediction(left, right, 1), right)
        np.testing.assert_array_equal(blend.blend_prediction(left, right, 0).view(np.uint64),
                                      left.view(np.uint64))
        for alpha in (.25, .5, .75):
            prediction = blend.blend_prediction(left, right, alpha)
            np.testing.assert_array_equal(prediction, (1-alpha)*left+alpha*right)
            self.assertFalse(np.array_equal(prediction, baseline+(1-alpha)*left+alpha*right))

    def test_interior_ensemble_SSE_contains_cross_terms_not_mean_endpoint_loss(self):
        y, baseline, labels, _, _ = example()
        w, hw = y+2., y-2.
        prediction = blend.blend_prediction(w, hw, .5)
        np.testing.assert_allclose(prediction, y, rtol=0, atol=2e-15)
        correct_loss = float(np.square(y-prediction).sum())
        mean_endpoint_loss = .5*float(np.square(y-w).sum())+.5*float(np.square(y-hw).sum())
        self.assertLess(correct_loss, 1e-25)
        self.assertGreater(mean_endpoint_loss, 100.)

    def test_all_25_cells_use_native_joint_XY_and_exact_whole_trial_wrong_pair_score(self):
        y, baseline, labels, w, hw = example(508012)
        bank = blend.score_session(y, baseline, labels, w, hw, lambda: None)
        denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
        baseline_sse = float(np.square(y-baseline).sum())
        self.assertEqual(bank["denominator"], denominator)
        self.assertEqual(bank["baseline_SSE"], baseline_sse)
        self.assertEqual(len(bank["cells"]), 25)
        for index, cell in enumerate(bank["cells"]):
            penalty_index, alpha_index = divmod(index, 5)
            alpha = blend.ALPHAS[alpha_index]
            self.assertEqual(cell["lambda"], LAMBDAS[penalty_index])
            self.assertEqual(cell["alpha"], alpha)
            prediction = w[penalty_index] if alpha == 0 else hw[penalty_index] if alpha == 1 else (
                (1-alpha)*w[penalty_index]+alpha*hw[penalty_index])
            sse = float(np.square(y-prediction).sum())
            wrong = direct_wrong(y, prediction, labels)
            self.assertEqual(cell["candidate_SSE"], sse)
            self.assertAlmostEqual(cell["wrong_pair_SSE"], wrong, places=10)
            self.assertAlmostEqual(cell["delta_R2"], (baseline_sse-sse)/denominator, places=13)
            self.assertAlmostEqual(cell["identity_increment"], (wrong-sse)/denominator, places=13)
            self.assertEqual(cell["exact_prediction_equals_components"]["W"],
                             [bool(np.array_equal(prediction, p)) for p in w])
            self.assertEqual(cell["exact_prediction_equals_components"]["HW"],
                             [bool(np.array_equal(prediction, p)) for p in hw])

    def test_outer_session_exclusion_and_exact_ties_choose_lambda_then_less_history(self):
        sessions = synthetic_banks()
        held = SESSIONS[0]
        for bank in sessions.values():
            for cell in bank["cells"]:
                cell["delta_R2"] = 0.
        for index, cell in enumerate(sessions[held]["cells"]):
            cell["delta_R2"] = 1e9*(25-index)
        selected, training, scores = blend.select_cell(sessions, held)
        self.assertEqual(selected, 20)  # all other sessions tie: lambda10,alpha0
        self.assertEqual(set(training), set(SESSIONS)-{held})
        self.assertTrue(all(value == 0 for value in scores))
        for session, bank in sessions.items():
            if session != held:
                bank["cells"][21]["delta_R2"] = 1.
                bank["cells"][23]["delta_R2"] = 1.
                bank["cells"][17]["delta_R2"] = 1.
        self.assertEqual(blend.select_cell(sessions, held)[0], 21)
        for session, bank in sessions.items():
            if session != held:
                bank["cells"][2]["delta_R2"] = 1.+1e-10
        self.assertEqual(blend.select_cell(sessions, held)[0], 2)

    def test_outer_tuning_weights_animals_equally_despite_unequal_training_session_counts(self):
        sessions = synthetic_banks()
        held = SESSIONS[0]
        self.assertEqual(ANIMALS[held], "Mihili")
        for session, bank in sessions.items():
            for cell in bank["cells"]:
                cell["delta_R2"] = -10.
            bank["cells"][0]["delta_R2"] = 2. if ANIMALS[session] == "Mihili" else 0.
            bank["cells"][1]["delta_R2"] = 0. if ANIMALS[session] == "Mihili" else 1.5
        selected, _, scores = blend.select_cell(sessions, held)
        self.assertEqual(selected, 0)
        self.assertEqual(scores[0], 1.)
        self.assertEqual(scores[1], .75)

    def test_complete_global_endpoint_duplicates_add_zero_and_interior_ensemble_adds_one(self):
        y, baseline, labels, _, _ = example(508013)
        perfect = np.ascontiguousarray(np.stack([y]*5))
        weak = np.ascontiguousarray(np.stack([baseline]*5))
        for w, hw, expected in ((perfect, weak, ["W"]), (weak, perfect, ["HW"]),
                                 (perfect, perfect, ["W", "HW"])):
            sessions, anchors, panels = complete_family(y, baseline, labels, w, hw)
            result = blend.summarize_banks(sessions, anchors, panels)
            self.assertEqual(result["structural_duplicate_of"], expected)
            self.assertEqual(result["scientific_trial_count_increment"], 0)
            self.assertFalse(result["eligible_for_comparable_primary_incumbent"])
        for displacement in (1., 1e-8):
            w = np.ascontiguousarray(np.stack([y+displacement]*5))
            hw = np.ascontiguousarray(np.stack([y-displacement]*5))
            result = blend.summarize_banks(*complete_family(y, baseline, labels, w, hw))
            self.assertEqual(result["structural_duplicate_of"], [])
            self.assertEqual(result["scientific_trial_count_increment"], 1)
            self.assertTrue(result["eligible_for_comparable_primary_incumbent"])
            self.assertEqual(result["literal_null_control_count_increment"], 0)
            self.assertEqual(result["alternative_explanation_count_increment"], 0)
            self.assertEqual(result["neural_refits"], 0)
            self.assertFalse(result["promotion_authorized"])
            for row in result["blend"]["sessions"]:
                self.assertIn(row["alpha"], (.25, .5, .75))
                if displacement == 1.:
                    self.assertEqual(row["alpha"], .5)
                else:
                    # At 1e-8 the subtraction baseline_SSE-SSE can make
                    # distinct interior losses exactly tie in delta_R2.
                    # The declared tie rule may then select less history.
                    # Duplicate accounting still requires exact arrays,
                    # even when allclose would identify both endpoints.
                    prediction = blend.blend_prediction(w[row["lambda_index"]],
                                                        hw[row["lambda_index"]], row["alpha"])
                    for endpoint in (w[row["lambda_index"]], hw[row["lambda_index"]]):
                        self.assertTrue(np.allclose(prediction, endpoint))
                        self.assertFalse(np.array_equal(prediction, endpoint))

    def test_anchor_contrasts_use_direct_common_SSE_and_both_preserved_penalty_conventions(self):
        sessions, anchors, panels = complete_family(*example(508014))
        result = blend.summarize_banks(sessions, anchors, panels)
        self.assertEqual(set(result["reference_contrasts"]), {
            name+"_"+kind for name in blend.COMPONENTS
            for kind in ("independently_selected", "blend_lambda_matched")})
        original = {name: {row["session_id"]: row for row in panels[name]["sessions"]}
                    for name in blend.COMPONENTS}
        for selected in result["blend"]["sessions"]:
            session = selected["session_id"]
            for name in blend.COMPONENTS:
                for kind, anchor in (("independently_selected", original[name][session]),
                    ("blend_lambda_matched", anchors[name][session]["lambda_rows"][selected["lambda_index"]])):
                    contrast = next(row for row in result["reference_contrasts"][name+"_"+kind]["sessions"]
                                    if row["session_id"] == session)
                    expected = (anchor["candidate_SSE"]-selected["candidate_SSE"])/sessions[session]["denominator"]
                    self.assertEqual(contrast["delta_R2"], expected)
                    self.assertEqual(contrast["identity_increment"], selected["identity_increment"]-anchor["identity_increment"])
                    self.assertEqual(contrast["reference_lambda"], anchor["lambda"])

    def test_original_selector_is_recomputed_and_original_scoring_support_cannot_change(self):
        sessions, anchors, panels = complete_family(*example(508015))
        replay = blend.validate_original_selections(anchors["W"], panels["W"])
        self.assertEqual(set(replay), set(SESSIONS))
        for field, value in (("lambda_index", 100), ("outer_training_sessions", list(SESSIONS)),
                             ("candidate_SSE", -1.)):
            altered = copy.deepcopy(panels["W"])
            altered["sessions"][0][field] = value
            with self.assertRaises(ValueError):
                blend.validate_original_selections(anchors["W"], altered)
        altered = copy.deepcopy(sessions)
        altered[SESSIONS[0]]["denominator"] += 1.
        with self.assertRaises(ValueError):
            blend.summarize_banks(altered, anchors, panels)
        incomplete = copy.deepcopy(sessions)
        incomplete.pop(SESSIONS[-1])
        with self.assertRaises(ValueError):
            blend.summarize_banks(incomplete, anchors, panels)

    def test_endpoint_scores_replay_and_direction_trial_grouping_is_required(self):
        y, baseline, labels, w, hw = example(508016)
        bank = blend.score_session(y, baseline, labels, w, hw, lambda: None)
        for name, values, alpha_index in (("W", w, 0), ("HW", hw, 4)):
            original = score_bank(y, labels, baseline, values, blend.COMPONENTS[name])
            for index in range(5):
                cell, expected = bank["cells"][index*5+alpha_index], original["lambda_rows"][index]
                self.assertTrue(all(cell[key] == expected[key] for key in blend.SCORE_KEYS))
        for broken_labels in (np.zeros_like(labels), labels[:-1]):
            with self.assertRaises(ValueError):
                blend.score_session(y, baseline, broken_labels, w, hw)

    def test_saved_component_layout_replay_and_actual_runner_boundary(self):
        values = example(508017)
        with tempfile.TemporaryDirectory() as directory:
            roots, expected_ids = saved_components(directory, values)
            y, baseline, labels, ids, predictions, old, source = runner.load_components(
                roots["W"], roots["HW"], SESSIONS[0])
            for loaded, original in ((y, values[0]), (baseline, values[1])):
                np.testing.assert_array_equal(loaded, original)
                self.assertEqual(loaded.strides, original.strides)
                self.assertFalse(loaded.flags.c_contiguous)
            np.testing.assert_array_equal(ids, expected_ids)
            self.assertEqual(source["component_input_offsets"], {"W": [0, 14], "HW": [-3, 14]})
            self.assertFalse(source["provider_source_values_read"])
            bank = blend.score_session(y, baseline, labels, predictions["W"], predictions["HW"])
            bank["component_score_rows"] = {name: old[name]["lambda_rows"] for name in blend.COMPONENTS}
            item = {"status": "complete_history_blend_session_bank", "session_id": SESSIONS[0],
                    "exact_original_component_arrays_support_replay": True,
                    "exact_original_component_scores_replay": True,
                    "provider_source_values_read": False, "neural_refits": 0,
                    "scientific_trial_count_increment": 0, "bank": bank, "source": source}
            runner.validate_session(item, SESSIONS[0])
            broken = copy.deepcopy(item)
            broken["bank"]["component_score_rows"]["HW"][0]["candidate_SSE"] += 1.
            with self.assertRaises(ValueError):
                runner.validate_session(broken, SESSIONS[0])

    def test_saved_components_reject_changed_original_support_and_stale_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            for mutation in ("baseline", "trial_order", "channel_support", "window"):
                roots, _ = saved_components(directory)
                folder = roots["HW"]/SESSIONS[0]
                if mutation in ("baseline", "trial_order"):
                    path = folder/(blend.COMPONENTS["HW"]+"_predictions.npz")
                    with np.load(path, allow_pickle=False) as saved:
                        arrays = {key: saved[key] for key in saved.files}
                    if mutation == "baseline":
                        arrays["baseline_xy"][0, 0, 0] += 1.
                    else:
                        arrays["trial_ids"] = arrays["trial_ids"][::-1]
                    np.savez(path, **arrays)
                else:
                    path = folder/"score_banks.json"
                    item = json.loads(path.read_text())
                    if mutation == "channel_support":
                        item["source"]["common_LMP_electrodes"] = [1, 3, 8]
                        item["banks"][blend.COMPONENTS["HW"]]["physical_electrode_ids"] = [1, 3, 8]
                    else:
                        item["source"]["native_target_offsets"] = [-3, 14]
                    path.write_text(json.dumps(item))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    runner.load_components(roots["W"], roots["HW"], SESSIONS[0])


if __name__ == "__main__":
    unittest.main()
