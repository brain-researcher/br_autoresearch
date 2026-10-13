"""Source-free fixtures for preserved PLS projections with new true-target heads."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import extract_primary_history_pls_basis as extractor
import primary_history_pls as original
import primary_history_pls_basis as model
import run_primary_history_pls_basis as runner
import run_primary_history_pls_basis_panel as wrapper
from primary_capacity import score_bank
from primary_capacity_matched_surrogate import average_banks
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_history_blend import restore_target_layout
from test_primary_elapsed_only import direct_wrong_pair_sse, example
from test_primary_history_pls import direct_heads


def fold_example(identity=False):
    rng = np.random.default_rng(195101)
    x = rng.normal(size=(5, 15, 6))
    x -= x.mean(axis=(0, 1))
    held = rng.normal(size=(2, 15, 6))
    u = x.reshape(-1, 6)[:, :2] + .3*rng.normal(size=(75, 2))
    u -= u.mean(axis=0)
    maps = [np.arange(5) if identity else np.roll(np.arange(5), r % 5) for r in range(8)]
    p, f, details = original.fit_fold_arms(x, held, u, maps)
    return x, held, u, maps, p, f, details


def flags():
    return ({s: {name: [False]*8 for name in model.SINGLE_REFERENCES} for s in SESSIONS},
            {s: {name: [False]*8 for name in model.ENSEMBLE_REFERENCES} for s in SESSIONS})


class HistoryPLSBasisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, y, cls.labels, cls.ids, cls.electrodes = example(195102, equal_counts=True)
        cls.y = restore_target_layout(np.asarray(y, dtype=np.float64))
        cls.old, cls.baseline, cls.p, cls.f = original.fit_banks(
            cls.x, cls.y, cls.labels, cls.ids, "example", electrode_ids=cls.electrodes)
        with patch.object(original, "fit_pls", side_effect=AssertionError("Projection refit forbidden")):
            cls.bank, cls.replayed_baseline, cls.b = model.fit_banks(
                cls.x, cls.y, cls.labels, cls.ids, "example", cls.old, cls.p, cls.f,
                electrode_ids=cls.electrodes)

    def test_identity_maps_replay_P_F_B_literally_without_any_projection_fit(self):
        x, held, u, maps, p, f, details = fold_example(identity=True)
        frozen = copy.deepcopy(details)
        with patch.object(original, "fit_pls", side_effect=AssertionError("Projection refit forbidden")):
            rp, rf, b, info = model.replay_fold_heads(x, held, u, maps, details)
        np.testing.assert_array_equal(rp, p)
        np.testing.assert_array_equal(rf, f)
        for r in range(8):
            np.testing.assert_array_equal(b[r], p)
            np.testing.assert_array_equal(f[r], p)
        self.assertEqual(details, frozen)
        self.assertEqual(info["new_projection_fits"], 0)
        self.assertEqual(info["stored_transforms_reused"], 9)

    def test_nonidentity_draw_changes_only_the_head_and_matches_independent_solves(self):
        x, held, u, maps, p, f, details = fold_example()
        frozen = copy.deepcopy(details)
        with patch.object(original, "fit_pls", side_effect=AssertionError("Projection refit forbidden")):
            rp, rf, b, info = model.replay_fold_heads(x, held, u, maps, details)
        np.testing.assert_array_equal(rp, p)
        np.testing.assert_array_equal(rf, f)
        for r, state in enumerate(details["control_transforms"]):
            z = original.apply_pls(x.reshape(-1, 6), state)
            evaluation = original.apply_pls(held.reshape(-1, 6), state)
            expected = direct_heads(z, evaluation, u)
            np.testing.assert_allclose(b[r], expected, atol=3e-12, rtol=3e-11)
        self.assertGreater(float(np.max(np.abs(b[1]-f[1]))), 1e-3)
        self.assertGreater(float(np.max(np.abs(b[1]-p))), 1e-3)
        self.assertEqual(details, frozen)
        self.assertTrue(info["basis_and_full_target_score_Grams_identical"])
        self.assertFalse(info["candidate_and_basis_capacity_matched"])

    def test_tampered_transform_covariance_and_trial_map_are_rejected(self):
        x, held, u, maps, _, _, details = fold_example()
        for alteration in ("mean", "weight", "covariance"):
            changed = copy.deepcopy(details)
            if alteration == "mean":
                changed["control_transforms"][2]["score_mean"][0] += 1.
            elif alteration == "weight":
                changed["control_transforms"][2]["components"][0]["weight"][0] = float("nan")
            else:
                changed["control_score_covariances"][2]["trace"] += 1.
            with self.subTest(alteration=alteration), self.assertRaises((ValueError, ArithmeticError)):
                model.replay_fold_heads(x, held, u, maps, changed)
        bad = maps.copy()
        bad[2] = np.array([0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            model.replay_fold_heads(x, held, u, bad, details)

    def test_original_zero_and_one_component_states_are_applied_without_rescue(self):
        u = np.arange(90, dtype=float).reshape(45, 2)
        u -= u.mean(axis=0)
        for k in (0, 1):
            x = np.zeros((3, 15, 4))
            if k:
                x[:, :, 0] = u[:, 0].reshape(3, 15)
            held = x[:1].copy()
            maps = [np.arange(3)]*8
            p, f, details = original.fit_fold_arms(x, held, u, maps)
            self.assertEqual(details["candidate_transform"]["retained_components"], k)
            rp, rf, b, _ = model.replay_fold_heads(x, held, u, maps, details)
            np.testing.assert_array_equal(rp, p)
            np.testing.assert_array_equal(rf, f)
            for r in range(8):
                np.testing.assert_array_equal(b[r], p)
            if k == 0:
                np.testing.assert_array_equal(b, 0.)

    def test_complete_HW_replay_native_scores_and_mean_losses_not_mean_predictions(self):
        np.testing.assert_array_equal(self.replayed_baseline, self.baseline)
        self.assertEqual(self.bank["new_projection_fits"], 0)
        self.assertEqual(self.bank["stored_transforms_reused"], 72)
        runner.validate_score_bank(self.bank)
        for index, row in enumerate(self.bank["lambda_rows"]):
            sse = float(np.mean([np.square(self.y-self.b[r, index]).sum() for r in range(8)]))
            wrong = float(np.mean([direct_wrong_pair_sse(self.y, self.b[r, index], self.labels) for r in range(8)]))
            self.assertAlmostEqual(row["candidate_SSE"], sse, places=8)
            self.assertAlmostEqual(row["wrong_pair_SSE"], wrong, places=8)
            self.assertAlmostEqual(row["delta_R2"], (self.bank["baseline_SSE"]-sse)/self.bank["denominator"], places=14)
        ensemble_sse = float(np.square(self.y-self.b[:, 0].mean(axis=0)).sum())
        self.assertGreater(self.bank["lambda_rows"][0]["candidate_SSE"]-ensemble_sse, 1e-5)
        # The actual runner replays every saved draw and its mean, including the
        # time-fast source Y/B layout after an ordinary C-layout archive read.
        restored_y = restore_target_layout(np.array(self.y, order="C"))
        restored_b = restore_target_layout(np.array(self.baseline, order="C"))
        runner.replay_scores(self.b, restored_y, restored_b, self.labels, self.bank,
                             model.RECIPE, seed_key="surrogate_seed_rows")

    def test_changed_original_prediction_or_HW_state_cannot_be_repaired_by_refitting(self):
        for which in ("P", "F", "HW"):
            p, f, banks = self.p.copy(), self.f.copy(), copy.deepcopy(self.old)
            if which == "P":
                p[4, 0, 0, 0] += 1.
            elif which == "F":
                f[7, 4, 0, 0, 0] += 1.
            else:
                banks[original.RECIPES[0]]["direction_fit_metadata"][0]["paired_weighted_covariance"]["trace"] += 1.
            with self.subTest(which=which), self.assertRaises(ValueError):
                model.fit_banks(self.x, self.y, self.labels, self.ids, "example", banks, p, f,
                                electrode_ids=self.electrodes)

    def test_counting_requires_global_literal_single_or_entire_aligned_ensemble_identity(self):
        single, ensemble = flags()
        result = model.duplicate_accounting(single, ensemble)
        self.assertEqual(result["scientific_trial_count_increment"], 1)
        self.assertEqual(result["alternative_explanation_count_increment"], 1)
        self.assertEqual(result["candidate_trial_count_increment"], 0)
        self.assertEqual(result["literal_null_control_count_increment"], 0)
        self.assertFalse(result["updates_primary_incumbent"])
        for s in SESSIONS:
            single[s]["PLS"] = [True]*8
        self.assertEqual(model.duplicate_accounting(single, ensemble)["scientific_trial_count_increment"], 0)
        single[SESSIONS[-1]]["PLS"][-1] = False
        self.assertEqual(model.duplicate_accounting(single, ensemble)["scientific_trial_count_increment"], 1)
        for s in SESSIONS:
            ensemble[s]["PLS_full_target"] = [True]*8
        self.assertEqual(model.duplicate_accounting(single, ensemble)["exact_duplicate_ensembles"], ["PLS_full_target"])
        ensemble[SESSIONS[-1]]["PLS_full_target"][-1] = False
        self.assertEqual(model.duplicate_accounting(single, ensemble)["scientific_trial_count_increment"], 1)
        # Numeric equal scores, fractions and truthy objects are not literal
        # prediction identity, nor are mixed per-session reference identities.
        single, ensemble = flags()
        for i, s in enumerate(SESSIONS):
            single[s]["HW" if i % 2 else "PLS"] = [True]*8
        self.assertEqual(model.duplicate_accounting(single, ensemble)["scientific_trial_count_increment"], 1)
        single[SESSIONS[0]]["HW"][0] = 1
        with self.assertRaises(ValueError):
            model.duplicate_accounting(single, ensemble)
        del single[SESSIONS[0]]
        with self.assertRaises(ValueError):
            model.duplicate_accounting(single, ensemble)

    def summary_fixture(self):
        def row(index, gain):
            return {"lambda": LAMBDAS[index], "delta_R2": gain, "identity_increment": .02,
                "candidate_SSE": 5.-10.*gain, "wrong_pair_SSE": 5.2-10.*gain,
                "candidate_R2": .5+gain}
        base = {"recipe": model.RECIPE, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8}
        banks, old_p, old_f, p, f = {}, {}, {}, {}, {}
        anchors = {"HW": {}, "blend": {}}
        for s in SESSIONS:
            b_seeds = [{**base, "lambda_rows": [row(i, (.2 if i == 0 else .006)+r*.0001) for i in range(5)]}
                       for r in range(8)]
            f_seeds = [{**base, "lambda_rows": [row(i, .001+r*.0001) for i in range(5)]} for r in range(8)]
            banks[s] = average_banks(b_seeds)
            banks[s]["direction_fit_metadata"] = self.bank["direction_fit_metadata"]
            old_p[s] = {**base, "lambda_rows": [row(i, .01) for i in range(5)]}
            old_f[s] = average_banks(f_seeds)
            label = {"session_id": s, "animal": ANIMALS[s], "lambda_index": 4,
                "baseline_R2": .5, "outer_training_sessions": [other for other in SESSIONS if other != s]}
            p[s] = {**label, **old_p[s]["lambda_rows"][4]}
            f[s] = {**label, **old_f[s]["lambda_rows"][4]}
            for name, index in (("HW", 1), ("blend", 2)):
                anchors[name][s] = {**label, "lambda_index": index, **row(index, .025)}
        return model.summarize_banks(banks, old_p, old_f, p, f, anchors)

    def test_old_P_selector_is_preserved_and_two_conditional_contrasts_stay_separate(self):
        report = self.summary_fixture()
        self.assertTrue(all(row["lambda"] == 10 and row["lambda_index"] == 4 for row in report["basis_only"]["sessions"]))
        self.assertTrue(all(row["lambda_index"] == 1 for row in report["original_anchors"]["HW"]["sessions"]))
        self.assertAlmostEqual(report["basis_only"]["primary_increment"], .00635, places=14)
        self.assertAlmostEqual(report["reference_contrasts"]["paired_PLS_minus_basis_only"]["primary_increment"], .00365, places=14)
        self.assertAlmostEqual(report["reference_contrasts"]["basis_only_minus_full_target"]["primary_increment"], .005, places=14)
        for r in range(8):
            self.assertAlmostEqual(report["each_basis_minus_matched_full_target"][r]["primary_increment"], .005, places=14)
        self.assertFalse(report["new_selector_fitted"])
        self.assertFalse(report["promotion_authorized"])

    def test_compact_export_excludes_nested_banks_and_requires_complete_replayed_panel(self):
        report = self.summary_fixture()
        single, ensemble = flags()
        report.update(model.duplicate_accounting(single, ensemble), status=runner.PANEL_STATUS,
            single_duplicate_reference_names=list(model.SINGLE_REFERENCES),
            ensemble_duplicate_reference_names=list(model.ENSEMBLE_REFERENCES),
            **{name: True for name in runner.REPLAY_FLAGS})
        for rows in [report["basis_only"], *report["original_anchors"].values()]:
            rows["sessions"][0]["nested_training_bank_DO_NOT_EXPORT"] = {"predictions": [1]*10000}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"summary.json"
            path.write_text(json.dumps(report))
            compact = extractor.extract(path)
            rendered = json.dumps(compact, indent=2, allow_nan=False)
            self.assertLessEqual(len(rendered.encode()), extractor.MAX_EXPORT_BYTES)
            self.assertNotIn("nested_training_bank_DO_NOT_EXPORT", rendered)
            self.assertNotIn("outer_training_sessions", rendered)
            self.assertNotIn("control_transforms", rendered)
            self.assertEqual(compact["scientific_trial_count_increment"], 1)
            self.assertEqual(len(compact["paired_PLS_minus_each_basis"]), 8)
            report[runner.REPLAY_FLAGS[-1]] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                extractor.extract(path)

    def test_deadline_during_final_summary_validation_never_releases_complete_effects(self):
        # Simulate fifteen already settled child results; no child, source or
        # scheduler is used. Parsing the terminal summary consumes the bound.
        clock = [0.]
        summary = {"status": runner.PANEL_STATUS, "scientific_trial_count_increment": 1,
            "promotion_authorized": False, "literal_null_control_count_increment": 0,
            "updates_primary_incumbent": False, **{name: True for name in runner.REPLAY_FLAGS}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(**{name: root/name for name in runner.SUMMARY_OPTIONS},
                out=root/"wrapper", summary_out=root/"summary", workers=1, max_seconds=1800)
            reads = []
            def read(path):
                reads.append(path)
                if len(reads) == 1:
                    return {"elapsed_seconds": 1.}
                clock[0] = 1801.
                return summary
            with patch.dict(wrapper.os.environ, {"SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "fixture"}), \
                 patch.object(wrapper, "read_json", side_effect=read), \
                 patch.object(wrapper, "validate_session"), \
                 patch.object(wrapper, "run_sessions", return_value=[{"session_id": s} for s in SESSIONS if s != wrapper.PILOT]), \
                 patch.object(wrapper.subprocess, "run") as launch, \
                 patch.object(wrapper.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(wrapper.signal, "signal"), patch.object(wrapper.signal, "alarm"):
                with self.assertRaises(TimeoutError):
                    wrapper.run_remainder(args)
            report = json.loads((args.out/"remaining_panel.json").read_text())
            self.assertEqual(report["status"], "technical_failure")
            self.assertFalse(report["scientific_effects_released"])
            self.assertEqual(report["scientific_trial_count_increment"], 0)
            self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
