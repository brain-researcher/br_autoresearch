"""Independent checks for native two-band acquisition, selection and decoding."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import high_power_acquired_reader as reader
import high_power_correspondence as selector
import high_power_correspondence_experiment as experiment
import high_power_selected_rank as selected_rank
import development_high_power_correspondence as runner
import finalize_high_power_correspondence as finalizer
from capacity_training import fit_capacity
from initializer_target_label_control import within_direction_map


def initial_design():
    labels = np.repeat(np.arange(8), 4)
    slots = (labels+np.tile(np.arange(4), 8)) % 4
    folds = [(np.flatnonzero(slots != fold), np.flatnonzero(slots == fold)) for fold in range(4)]
    return labels, folds


def completed_fit_packet():
    """Synthetic saved-fit packet; no source identities or scores are opened."""
    _, folds = initial_design()
    pools = [list(range(12*d, 12*d+12)) for d in range(8)]
    initial = [trial for pool in pools for trial in pool[:4]]
    roles = {"physical_channels": list(range(1, 97)), "initial_trial_ids": initial,
             "fitting_trial_ids": list(range(96)), "evaluation_trial_ids": list(range(96, 136)),
             "initial_inner_folds": [{"fitting_trials": [initial[i] for i in fit],
                                      "validation_trials": [initial[i] for i in valid]} for fit, valid in folds]}
    def panel():
        cells = [{"training_trials": n, "electrodes": e}
                 for n in reader.CHECKPOINTS for e in (8, 16, 32, 96)]
        return {"cells": cells, "fitted_models": copy.deepcopy(cells)}
    sessions = [{"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                 "all95_target_models_fixed_before_evaluation": True,
                 "all30_target_orders_fixed_before_post_initial_reads": True,
                 "target_unit_count": 10, "source_roles": copy.deepcopy(roles),
                 "panels": {selector.POLICY: panel(), selector.CONTROL: panel()},
                 "seed_panels": [{"mc_seed": seed, "panel": panel()} for seed in selector.MC_SEEDS],
                 "selection_records": [{"mc_seed": seed, "view": view}
                                       for seed in selector.BRANCHES for view in selector.VIEWS]}
                for filename in runner.DEVELOPMENT]
    return {"status": "technical_failure_high_power_correspondence_incomplete",
        "stage": "post_fit_reference_comparison", "error": "KeyError: 'null_SSE'",
        "recipe": "outputs/high_power_correspondence_proposal.md", "analysis_id": "development_search",
        "policy": selector.POLICY, "control": selector.CONTROL, "MC_seeds": list(selector.MC_SEEDS),
        "held_session_accessed": False, "held_openings_charged": 0, "valid_evaluations_charged": 0,
        "named_explanatory_amendment_activated": False, "raw_arrays_or_decoder_coefficients_written": False,
        "decoder_target_labels_shuffled": False, "input_coordinates_per_electrode": 2,
        "native_power_guide_pairs": [[100., 200.], [200., 400.]], "LMP_neural_values_read": False,
        "LMP_columns_zero_padded": False,
        "fit_progress": {"outer_training_refits": 760, "inner_ridge_fits": 15200,
                         "capacity_inner_candidate_scores": 76000, "selector_orderings": 240,
                         "direction_permutation_maps": 1600, "direction_only_models": 40},
        "fitting_trial_read_calls": 9984, "evaluation_trials_read": 320,
        "completed_session_ids": list(runner.DEVELOPMENT), "sessions": sessions,
        "elapsed_seconds": 123.5, "process_CPU_seconds": 102.25, "maximum_RSS_KiB": 64000,
        "logical_neural_bytes": 120000, "unique_logical_neural_bytes": 60000, "retained_report_bytes": 1234}


class HighPowerTest(unittest.TestCase):
    def test_reader_never_reads_LMP_and_counts_actual_two_band_dtype(self):
        guide = np.array([[channel, lo, hi] for channel in (1, 2)
                          for lo, hi in ((0, 0), (100, 200), (200, 400))], dtype=float)
        units = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
        lfp = np.full((6, 10), np.nan, dtype=np.float32)
        for row in (1, 2, 4, 5):
            lfp[row] = 10*row+np.arange(1, 11)
        spikes = np.vstack((np.arange(10), np.full(10, .5), 2*np.arange(10))).astype(np.float64)
        selections = []

        class NeuralNode:
            shape = lfp.shape
            def __getitem__(self, key):
                physical_rows, interval = key
                positions = np.asarray(physical_rows).tolist()
                if not set(positions) <= {1, 2, 4, 5}:
                    raise AssertionError("LMP numeric values must never be read")
                selections.append(positions)
                return lfp[key]

        class Group:
            pass

        group = Group()
        class Handle:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter(("trial_data",))
            def __getitem__(self, _): return group

        fields = {"M1_lfp_guide": guide.T, "M1_unit_guide": units.T,
                  "M1_lfp": NeuralNode(), "M1_spikes": spikes}
        filename = "Mihili_CO_20140217.mat"
        source = {"filename": filename, "session_role": "development", "common_physical_channels": [1, 2]}
        role = {"filename": filename, "session_role": "development", "evaluation_trials": [1],
                "nested_training_sets": {"96": {"training_trials": [0]}}}
        records = {0: {"window_start_zero_based": 0, "window_end_exclusive": 10,
                       "array_shapes": {"M1_lfp": [10, 6], "M1_spikes": [10, 3]}}}
        report = defaultdict(int)
        exposure = reader.Exposure(report)
        with patch.object(reader.h5py, "File", return_value=Handle()), \
                patch.object(reader.h5py, "Group", Group), \
                patch.object(reader, "mclass", return_value="struct"), \
                patch.object(reader, "field", side_effect=lambda handle, group, name, trial: fields[name]):
            x, y, reference = reader.read_trials(Path(filename), source, role, records,
                                                [0], [2, 1], lambda: None, exposure)
            reader.read_trials(Path(filename), source, role, records, [0], [2], lambda: None,
                               exposure, reference)
        np.testing.assert_array_equal(x, [[45.5, 55.5, 15.5, 25.5]])
        np.testing.assert_array_equal(y, [[4.5, .5, 9.]])
        self.assertEqual(selections, [[1, 2, 4, 5], [4, 5]])
        # Two bands, ten bins, native float32; repeated target/source reads
        # count again in the read sum but never in unique logical elements.
        self.assertEqual(report["logical_neural_bytes"], 40*4+30*8+20*4+30*8)
        self.assertEqual(report["unique_logical_neural_bytes"], 40*4+30*8)
        self.assertEqual(report["fitting_trial_read_calls"], 2)

    def test_selected_trajectory_retains_two_columns_and_reads_only_frozen_channels(self):
        pools = [list(range(12*d, 12*d+12)) for d in range(8)]
        role = {"fitting_pool_by_direction": pools}
        channels = list(range(1, 97))
        trajectory = reader.Trajectory(Path("unused"), {"common_physical_channels": channels},
            role, {}, [96, 2], lambda: None, object())
        x, y = np.arange(32*192).reshape(32, 192), np.arange(32*3).reshape(32, 3)
        guide = object()
        trajectory.initialize_selected(x, y, channels, guide)
        observed_x, observed_y, ids = trajectory.observed()
        np.testing.assert_array_equal(observed_x, x[:, [190, 191, 2, 3]])
        np.testing.assert_array_equal(observed_y, y)
        self.assertEqual(ids, [trial for pool in pools for trial in pool[:4]])
        expected_new = [trial for pool in pools for trial in pool[4:6]]
        with patch.object(reader, "read_trials", return_value=(np.zeros((16, 4)), np.zeros((16, 3)), guide)) as read:
            trajectory.acquire_to(48)
        self.assertEqual(read.call_args.args[4:6], (expected_new, [96, 2]))
        self.assertIs(read.call_args.args[-1], guide)
        self.assertEqual(trajectory.observed()[0].shape, (48, 4))
        with self.assertRaises(ValueError):
            trajectory.acquire_to(80)

    def test_two_by_two_gain_matches_direct_objective_and_original_maps(self):
        rng = np.random.default_rng(58001)
        labels, _ = initial_design()
        x, y = rng.normal(size=(32, 10)), rng.normal(size=(32, 3))
        x[:, 1] += 1.3*x[:, 0]
        ids, channels = np.arange(32)*13+100, [1, 2, 3, 4, 5]
        dx = x-np.stack([x[labels == d].mean(axis=0) for d in range(8)])[labels]
        dxsd = dx.std(axis=0)
        dxsd[dxsd == 0] = 1
        xx = (dx-dx.mean(axis=0))/dxsd
        ysd = y.std(axis=0)
        ysd[ysd == 0] = 1
        yy = (y-np.stack([y[labels == d].mean(axis=0) for d in range(8)])[labels])/ysd
        yy -= yy.mean(axis=0)
        for replicate in (None, 0, 4):
            if replicate is None:
                mapping, expected_maps = np.arange(32), []
            else:
                mapping, expected_maps = within_direction_map(ids, labels, "example", "outer", replicate)
            expected = []
            for electrode in range(5):
                block = xx[:, 2*electrode:2*electrode+2]
                gram, cross = block.T@block/32, block.T@yy[mapping]/32
                expected.append(np.trace(cross.T@np.linalg.solve(np.eye(2)+gram, cross))/3)
            expected = np.asarray(expected)
            order, scores, maps = selector.rank_view(x, y, labels, ids, channels, "example", "outer", replicate)
            np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(order, np.lexsort((channels, -expected)))
            self.assertEqual(maps, expected_maps)

    def test_control_maps_accept_identity_without_target_or_channel_mutation(self):
        rng = np.random.default_rng(58002)
        labels, _ = initial_design()
        x, y = rng.normal(size=(32, 8)), rng.normal(size=(32, 2))
        old_x, old_y = x.copy(), y.copy()
        ids, channels = np.arange(32)*7, [1, 2, 3, 4]
        class Identity:
            def seed(self, literal, version): self.version = version
            def shuffle(self, values): pass
        true = selector.rank_view(x, y, labels, ids, channels, "example", "outer", None)
        with patch("initializer_target_label_control.random.Random", side_effect=Identity):
            control = selector.rank_view(x, y, labels, ids, channels, "example", "outer", 3)
        np.testing.assert_array_equal(control[0], true[0])
        np.testing.assert_array_equal(control[1], true[1])
        self.assertEqual(len(control[2]), 8)
        for direction in control[2]:
            self.assertEqual(direction["recipient_trial_ids"], direction["donor_trial_ids"])
        np.testing.assert_array_equal(x, old_x)
        np.testing.assert_array_equal(y, old_y)
        tied, gains, _ = selector.rank_view(np.zeros((32, 8)), np.zeros((32, 2)),
            labels, ids, channels, "example", "outer")
        np.testing.assert_array_equal(tied, np.arange(4))
        np.testing.assert_array_equal(gains, np.zeros(4))

    def test_actual_two_E_indexing_and_legal_rank_match_unpadded_decoder(self):
        rng = np.random.default_rng(58003)
        labels, folds = initial_design()
        x, y = rng.normal(size=(32, 192)), rng.normal(size=(32, 10))
        order = np.arange(96)[::-1]
        fitted, columns = selected_rank.fit_selected_rank(x, y, labels, folds,
            [order.copy() for _ in folds], order, 2, defaultdict(int))
        np.testing.assert_array_equal(columns, [190, 191, 188, 189])
        self.assertEqual(fitted.coefficients.shape, (4, 10))
        self.assertEqual(fitted.legal_ranks, (1, 2, 4, "full"))
        self.assertEqual(fitted.candidate_count, 20)
        direct = fit_capacity(x[:, [190, 191, 188, 189]], y, labels, folds, "reduced_rank", defaultdict(int))
        self.assertEqual((fitted.rank, fitted.alpha), (direct.rank, direct.alpha))
        np.testing.assert_allclose(fitted.coefficients, direct.coefficients, rtol=1e-12, atol=1e-13)

    def test_each_inner_view_excludes_its_validation_X_and_Y_from_all_orders(self):
        rng = np.random.default_rng(58005)
        labels, folds = initial_design()
        x, y = rng.normal(size=(32, 192)), rng.normal(size=(32, 10))
        ids, channels = np.arange(32)*11+100, list(range(1, 97))
        before, records = selector.order_bank(x, y, labels, ids, channels, folds, "example", defaultdict(int))
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += rng.normal(size=(8, 192))*1e5
        changed_y[folds[0][1]] -= rng.normal(size=(8, 10))*1e5
        after, changed = selector.order_bank(changed_x, changed_y, labels, ids, channels, folds,
                                             "example", defaultdict(int))
        for seed in selector.BRANCHES:
            np.testing.assert_array_equal(before[seed]["0"], after[seed]["0"])
            original = next(row for row in records if row["mc_seed"] == seed and row["view"] == "0")
            other = next(row for row in changed if row["mc_seed"] == seed and row["view"] == "0")
            self.assertEqual(original, other)

    def test_all_30_orders_precede_future_reads_and_all_heads_use_true_Y(self):
        rng = np.random.default_rng(58004)
        filename = "Mihili_CO_20140217.mat"
        pools = [list(range(12*d, 12*d+12)) for d in range(8)]
        ids = [trial for pool in pools for trial in pool[:4]]
        labels, folds = initial_design()
        role = {"filename": filename, "animal": "Mihili", "fitting_pool_by_direction": pools,
                "nested_training_sets": {"32": {"training_trials": ids, "inner_folds": [
                    {"fitting_trials": [ids[index] for index in fit],
                     "validation_trials": [ids[index] for index in valid]} for fit, valid in folds]}}}
        records = {trial: {"direction_index": direction} for direction, pool in enumerate(pools) for trial in pool}
        source = {"common_physical_channels": list(range(1, 97))}
        x, y = rng.normal(size=(32, 192)), rng.normal(size=(32, 10))
        true_y = y.copy()
        progress = defaultdict(int)
        calls = []
        case = self

        class Trajectory:
            def __init__(self, path, source, role, records, channels, check, exposure):
                self.channels = channels
                case.assertEqual(progress["selector_orderings"], 30)
            def initialize_selected(self, xx, yy, channels, guide):
                np.testing.assert_array_equal(yy, true_y)
                case.assertEqual(xx.shape, (32, 192))

        def model():
            progress["outer_training_refits"] += 1
            return SimpleNamespace(number=progress["outer_training_refits"])
        def full_fit(xx, yy, dd, ff, method, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("full_true_target")
            return model()
        def selected_fit(xx, yy, dd, ff, inner, outer, count, pp):
            np.testing.assert_array_equal(yy, true_y)
            calls.append("scarce_true_target")
            return model(), np.asarray([2*e+b for e in outer[:count] for b in range(2)])
        def later(trajectory, total, records, pp, make_null=False):
            case.assertEqual(progress["selector_orderings"], 30)
            calls.append("future_acquisition")
            return model(), object() if make_null else None

        with patch.object(experiment, "read_trials", return_value=(x, y, object())) as reads, \
                patch.object(experiment, "Trajectory", Trajectory), \
                patch.object(experiment, "fit_capacity", side_effect=full_fit), \
                patch.object(experiment, "fit_selected_rank", side_effect=selected_fit), \
                patch.object(experiment, "later_fit", side_effect=later):
            models, nulls, guide, selections = experiment.fit_target(Path(filename), source, role, records,
                progress, lambda: None, object())
        self.assertEqual(len(selections), 30)
        self.assertEqual(progress["direction_permutation_maps"], 200)
        self.assertEqual(progress["outer_training_refits"], 95)
        self.assertEqual(len(models), 120)
        self.assertEqual(len({id(item[-1]) for item in models}), 95)
        self.assertEqual(calls.count("scarce_true_target"), 18)
        self.assertEqual(calls.count("future_acquisition"), 76)
        self.assertEqual(reads.call_count, 1)
        np.testing.assert_array_equal(y, true_y)

    def test_runner_rejects_missing_model_or_order_before_evaluation(self):
        for model_count, order_count in ((94, 30), (95, 29)):
            with self.subTest(models=model_count, orders=order_count):
                self.check_missing_barrier(model_count, order_count)

    def check_missing_barrier(self, model_count, order_count):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                reference_report=root/"closed_initializer.json", lmp_reference=root/"closed_lmp.json",
                output_dir=root/"failed_attempt", max_seconds=270)
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

    def test_score_path_uses_two_columns_and_mean_seed_SSE_with_all20_cells(self):
        channels = list(range(1, 97))
        labels = np.repeat(np.arange(8), 5)
        x, y = np.tile(np.arange(192), (40, 1)), np.ones((40, 2))
        statistics = SimpleNamespace(target_mean=np.zeros(2), target_sd=np.ones(2))
        null = SimpleNamespace(direction_target_mean=np.zeros((8, 2)),
                               target_mean=np.zeros(2), target_sd=np.ones(2))
        models = []
        for branch in selector.BRANCHES:
            for total in reader.CHECKPOINTS:
                for count in (8, 16, 32, 96):
                    selected = tuple(channels[::-1][:count])
                    amplitude = 0 if branch is None or count == 96 else branch+1
                    fitted = SimpleNamespace(statistics=statistics, rank=1, alpha=1.,
                        candidate_count=25, amplitude=amplitude, channels=selected)
                    models.append((branch, total, count, selected, fitted))
        def prediction(model, xx, dd):
            columns = [2*(channel-1)+band for channel in model.channels for band in range(2)]
            np.testing.assert_array_equal(xx, x[:, columns])
            return y+model.amplitude
        with patch.object(experiment, "predict_capacity", side_effect=prediction):
            panels, seeds = experiment.score_target(models, {n: null for n in reader.CHECKPOINTS},
                                                     x, y, labels, channels)
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(panels[selector.CONTROL]["cells"]), 20)
        self.assertAlmostEqual(panels[selector.POLICY]["mean_Q"], 1.)
        self.assertAlmostEqual(panels[selector.CONTROL]["mean_Q"], -7.25)
        for cell in panels[selector.CONTROL]["cells"]:
            self.assertAlmostEqual(cell["standardized_SSE"], 0. if cell["electrodes"] == 96 else 880.)
            self.assertEqual(cell["LFP_numeric_elements"], cell["LFP_electrode_trials"]*20)
            self.assertEqual(cell["target_numeric_elements"], cell["training_trials"]*2*10)

    def test_historical_replay_accepts_changed_spectral_models_but_checks_targets_maps_and_new_full96(self):
        def panel(width, q, rank):
            cells = [{"training_trials": n, "electrodes": e, "null_SSE": 100.,
                "standardized_SSE": (1-q)*100, "standardized_wrong_SSE": 80.,
                "native_target_SSE": (1-q)*100, "Q": q, "Q_wrong": .2,
                "pairing_increment": q-.2, "direction_only_Q": .1,
                "delta_Q_from_direction_only": q-.1,
                "LFP_electrode_trials": 32*96+(n-32)*e,
                "LFP_numeric_elements": (32*96+(n-32)*e)*10*width,
                "input_coordinates_per_electrode": width, "target_numeric_elements": n*10*10}
                for n in reader.CHECKPOINTS for e in (8, 16, 32, 96)]
            return {"cells": cells, "fitted_models": [{**cell,
                "channels": list(range(1, cell["electrodes"]+1)),
                "rank": rank, "alpha": float(width), "candidate_count": 25} for cell in cells]}
        sessions, old, low = [], [], []
        for filename in runner.DEVELOPMENT:
            support = {"filename": filename,
                "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                "source_roles": {"filename": filename}, "target_unit_count": 10}
            maps = [{"mc_seed": seed, "view": view,
                     "target_label_permutations": [{"synthetic_map": [seed, view]}]}
                    for seed in selector.MC_SEEDS for view in selector.VIEWS]
            three_band = {"initial_marginal": panel(3, .4, 4), experiment.PREVIOUS_CONTROL: panel(3, .3, 2)}
            old.append({**support, "selection_records": copy.deepcopy(maps), "panels": three_band})
            low.append({**support, "selection_records": copy.deepcopy(maps),
                "panels": {**copy.deepcopy(three_band), experiment.LMP_POLICY: panel(1, .25, 1),
                           experiment.LMP_CONTROL: panel(1, .22, 1)}})
            sessions.append({**support, "selection_records": copy.deepcopy(maps),
                "panels": {selector.POLICY: panel(2, .35, 2), selector.CONTROL: panel(2, .35, 2)},
                "seed_panels": [{"mc_seed": seed, "panel": panel(2, .35, 2)} for seed in selector.MC_SEEDS]})
        previous = {"status": "completed_one_initializer_target_label_control", "held_session_accessed": False,
            "control": experiment.PREVIOUS_CONTROL, "MC_seeds": list(selector.MC_SEEDS), "sessions": old}
        lmp = {"status": "completed_two_LMP_correspondence_comparisons", "held_session_accessed": False,
            "policy": experiment.LMP_POLICY, "control": experiment.LMP_CONTROL,
            "MC_seeds": list(selector.MC_SEEDS), "recipe": "outputs/lmp_only_correspondence_proposal.md", "sessions": low}
        self.assertEqual(experiment.attach_references(copy.deepcopy(sessions), previous, lmp), (0., 0.))
        # Authentic older initial_marginal cells have no serialized SSE trio.
        # Preserve that absence; do not reconstruct a denominator from Q.
        legacy_previous, legacy_lmp = copy.deepcopy(previous), copy.deepcopy(lmp)
        for packet in (legacy_previous, legacy_lmp):
            for session in packet["sessions"]:
                legacy = session["panels"]["initial_marginal"]
                for row in legacy["cells"]+legacy["fitted_models"]:
                    for key in ("null_SSE", "standardized_SSE", "standardized_wrong_SSE"):
                        row.pop(key)
        self.assertEqual(experiment.attach_references(copy.deepcopy(sessions), legacy_previous, legacy_lmp), (0., 0.))
        for packet_name, policy in (("previous", experiment.PREVIOUS_CONTROL),
                                   ("lmp", experiment.LMP_POLICY), ("lmp", experiment.LMP_CONTROL)):
            bad_previous, bad_lmp = copy.deepcopy(legacy_previous), copy.deepcopy(legacy_lmp)
            packet = bad_previous if packet_name == "previous" else bad_lmp
            packet["sessions"][0]["panels"][policy]["cells"][0].pop("null_SSE")
            with self.subTest(required_policy=policy), self.assertRaisesRegex(ValueError, "Required control/LMP SSE"):
                experiment.attach_references(copy.deepcopy(sessions), bad_previous, bad_lmp)
        partial = copy.deepcopy(previous)
        partial["sessions"][0]["panels"]["initial_marginal"]["cells"][0].pop("null_SSE")
        with self.assertRaisesRegex(ValueError, "Partial legacy SSE"):
            experiment.attach_references(copy.deepcopy(sessions), partial, lmp)
        changed_legacy = copy.deepcopy(legacy_lmp)
        changed_legacy["sessions"][0]["panels"]["initial_marginal"]["cells"][0]["Q"] += .01
        with self.assertRaisesRegex(ValueError, "scientific records disagree"):
            experiment.attach_references(copy.deepcopy(sessions), legacy_previous, changed_legacy)
        # Cross-spectrum rank, penalty and Q deliberately differ above. Only
        # representation-independent historical quantities must replay.
        for key, value in (("null_SSE", 101.), ("direction_only_Q", .11),
                           ("target_numeric_elements", 999), ("LFP_numeric_elements", 999)):
            changed = copy.deepcopy(sessions)
            changed[0]["panels"][selector.POLICY]["fitted_models"][0][key] = value
            with self.subTest(field=key), self.assertRaises((ValueError, ArithmeticError)):
                experiment.attach_references(changed, previous, lmp)
        for key, value in (("rank", 4), ("Q", .36)):
            changed = copy.deepcopy(sessions)
            row = next(r for r in changed[0]["seed_panels"][0]["panel"]["fitted_models"] if r["electrodes"] == 96)
            row[key] = value
            with self.subTest(full96_field=key), self.assertRaises(ArithmeticError):
                experiment.attach_references(changed, previous, lmp)
        changed = copy.deepcopy(sessions)
        changed[0]["selection_records"][0]["target_label_permutations"] = []
        with self.assertRaisesRegex(ArithmeticError, "maps differ"):
            experiment.attach_references(changed, previous, lmp)

    def test_all_eight_sessions_finish_before_historical_outcome_reads(self):
        class ReferencesReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, roles = root/"metadata.json", root/"roles.json"
            metadata.write_text("{}")
            roles.write_text(json.dumps({"sessions": [{"filename": name} for name in runner.DEVELOPMENT]}))
            args = SimpleNamespace(source_root=root, metadata_report=metadata, trial_roles=roles,
                reference_report=root/"closed_initializer.json", lmp_reference=root/"closed_lmp.json",
                output_dir=root/"attempt", max_seconds=270)
            fitted, evaluated, read_paths = [], [], []
            original_read = Path.read_text
            def read_text(path, *pargs, **kwargs):
                read_paths.append(path)
                if path in (args.reference_report, args.lmp_reference):
                    self.assertEqual(fitted, list(runner.DEVELOPMENT))
                    self.assertEqual(evaluated, fitted)
                    self.assertEqual(path, args.reference_report)
                    raise ReferencesReached()
                return original_read(path, *pargs, **kwargs)
            def metadata_source(metadata, roles, filename):
                return ({"filename": filename, "common_physical_channels": list(range(1, 97))},
                    {"filename": filename, "animal": "Mihili" if filename.startswith("Mihili") else "Chewie",
                     "evaluation_trials": list(range(40))}, {i: {"direction_index": i//5} for i in range(40)})
            def fit(path, source, role, records, progress, check, exposure):
                self.assertNotIn(args.reference_report, read_paths)
                self.assertNotIn(args.lmp_reference, read_paths)
                fitted.append(path.name)
                for key, value in (("outer_training_refits", 95), ("inner_ridge_fits", 1900),
                        ("capacity_inner_candidate_scores", 9500), ("selector_orderings", 30),
                        ("direction_permutation_maps", 200)):
                    progress[key] += value
                exposure.report["fitting_trial_read_calls"] += 1248
                return [], {}, None, []
            def evaluation(path, source, role, records, ids, channels, check, exposure, guide, evaluation):
                self.assertTrue(evaluation)
                self.assertEqual(fitted[-1], path.name)
                evaluated.append(path.name)
                exposure.report["evaluation_trials_read"] += 40
                return np.zeros((40, 192)), np.ones((40, 10)), None
            with patch.object(runner.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(runner, "selected_metadata", side_effect=metadata_source), \
                    patch.object(runner, "fit_target", side_effect=fit), \
                    patch.object(runner, "read_trials", side_effect=evaluation), \
                    patch.object(runner, "score_target", return_value=({}, [])), \
                    patch.object(runner, "role_record", side_effect=lambda source, role: {"filename": role["filename"]}), \
                    patch.object(Path, "read_text", read_text):
                with self.assertRaises(ReferencesReached):
                    runner.main()
            self.assertEqual(len(fitted), 8)
            self.assertNotIn(args.lmp_reference, read_paths)

    def test_finalize_validator_accepts_only_complete_uncharged_known_failure(self):
        packet = completed_fit_packet()
        finalizer.validate_completed_fit(packet)
        for key, value in (("stage", "new_target_fits"), ("error", "Different failure"),
                           ("valid_evaluations_charged", 1), ("scientific_scores_released", True),
                           ("held_session_accessed", True), ("LMP_neural_values_read", True),
                           ("fitting_trial_read_calls", 9983), ("evaluation_trials_read", 319)):
            changed = copy.deepcopy(packet)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                finalizer.validate_completed_fit(changed)
        for key in finalizer.EXPECTED_FITS:
            changed = copy.deepcopy(packet)
            changed["fit_progress"][key] -= 1
            with self.subTest(count=key), self.assertRaises(ValueError):
                finalizer.validate_completed_fit(changed)
        changes = (
            lambda p: p["sessions"].pop(),
            lambda p: p["sessions"][0]["selection_records"].pop(),
            lambda p: p["sessions"][0]["seed_panels"].pop(),
            lambda p: p["sessions"][0]["panels"][selector.POLICY]["fitted_models"].pop(),
            lambda p: p["sessions"][0]["source_roles"]["evaluation_trial_ids"].__setitem__(0, 0),
            lambda p: p["sessions"][0]["source_roles"]["initial_inner_folds"][0]["fitting_trials"].pop(),
        )
        for index, mutate in enumerate(changes):
            changed = copy.deepcopy(packet)
            mutate(changed)
            with self.subTest(incomplete_support=index), self.assertRaises(ValueError):
                finalizer.validate_completed_fit(changed)

    def test_finalizer_uses_only_saved_JSON_preserves_fit_costs_and_fails_uncharged(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                saved, previous, lmp = root/"saved.json", root/"previous.json", root/"lmp.json"
                original = completed_fit_packet()
                saved_text = json.dumps(original)
                saved.write_text(saved_text)
                previous.write_text('{"reference_kind": "original"}')
                lmp.write_text('{"reference_kind": "LMP"}')
                args = SimpleNamespace(completed_fit_report=saved, reference_report=previous,
                    lmp_reference=lmp, output_dir=root/"fresh_finalization", max_seconds=60)
                read_paths = []
                original_read = Path.read_text
                def read_text(path, *pargs, **kwargs):
                    read_paths.append(path)
                    self.assertIn(path, (saved, previous, lmp))
                    return original_read(path, *pargs, **kwargs)
                def finish(report, prior, low):
                    self.assertEqual(prior, {"reference_kind": "original"})
                    self.assertEqual(low, {"reference_kind": "LMP"})
                    self.assertEqual(report["fit_progress"], original["fit_progress"])
                    self.assertEqual(report["new_target_fits"], 0)
                    self.assertEqual(report["new_scientific_source_reads"], 0)
                    if fail:
                        raise ValueError("Rejected historical reference")
                    report.update(status="completed_two_high_power_correspondence_comparisons",
                                  stage="complete", scientific_scores_released=True,
                                  valid_evaluations_charged=2, valid_evaluations_total=34)
                with patch.object(finalizer.argparse.ArgumentParser, "parse_args", return_value=args), \
                        patch.object(finalizer, "finalize_scientific_report", side_effect=finish), \
                        patch.object(experiment, "fit_target", side_effect=AssertionError("No refit")), \
                        patch.object(experiment, "read_trials", side_effect=AssertionError("No neural read")), \
                        patch.object(reader.h5py, "File", side_effect=AssertionError("No HDF5 source access")), \
                        patch.object(Path, "read_text", read_text):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "Rejected historical reference"):
                            finalizer.main()
                    else:
                        finalizer.main()
                self.assertEqual(read_paths, [saved, previous, lmp])
                self.assertEqual(saved.read_text(), saved_text)
                output = json.loads((args.output_dir/"high_power_correspondence.json").read_text())
                for key in ("elapsed_seconds", "process_CPU_seconds", "maximum_RSS_KiB", "fit_progress",
                            "logical_neural_bytes", "unique_logical_neural_bytes",
                            "fitting_trial_read_calls", "evaluation_trials_read"):
                    self.assertEqual(output[key], original[key])
                self.assertEqual(output["new_target_fits"], 0)
                self.assertEqual(output["new_scientific_source_reads"], 0)
                self.assertEqual(output["reused_fit_report_retained_bytes"], original["retained_report_bytes"])
                self.assertEqual(output["original_post_fit_failure"], original["error"])
                self.assertEqual(output["valid_evaluations_charged"], 0 if fail else 2)
                self.assertEqual(output["scientific_scores_released"], not fail)
                self.assertEqual(output["finalization_resources"]["new_target_fits"], 0)


if __name__ == "__main__":
    unittest.main()
