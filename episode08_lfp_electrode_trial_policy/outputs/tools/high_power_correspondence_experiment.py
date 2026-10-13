"""Two-band high-power correspondence; actual2E selection and true-target decoding."""
import copy

import numpy as np

from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from initializer_target_label_control import CONTROL as PREVIOUS_CONTROL
from high_power_acquired_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from high_power_correspondence import BRANCHES, CONTROL, MC_SEEDS, POLICY, mean_control_cells, order_bank
from high_power_selected_rank import fit_selected_rank
from prepare_development_trial_roles import DEVELOPMENT

from lmp_correspondence import POLICY as LMP_POLICY, CONTROL as LMP_CONTROL

ALL_POLICIES = (POLICY, CONTROL, "initial_marginal", PREVIOUS_CONTROL, LMP_POLICY, LMP_CONTROL)


def fit_target(path, source, role, records, progress, check, exposure):
    channels, ids = source["common_physical_channels"], balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original96-electrode/32-row initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    banks, selections = order_bank(x, y, labels, ids, channels, folds, role["filename"], progress, check)
    if len(selections) != 30 or set(banks) != set(BRANCHES):
        raise ArithmeticError("All30 orders required before future reads")
    models, nulls, shared = [], {32: fit_direction_null(y, labels)}, {}
    progress["direction_only_models"] += 1
    full = Trajectory(path, source, role, records, channels, check, exposure)
    full.initialize_selected(x, y, channels, guide)
    fitted = fit_capacity(x.copy(), y.copy(), labels.copy(), folds, "reduced_rank", progress)
    for total in CHECKPOINTS:
        check()
        if total > 32:
            fitted, null = later_fit(full, total, records, progress, True)
            nulls[total] = null
        shared[total] = fitted
    for seed in BRANCHES:
        orders = banks[seed]
        inner, outer = [orders[str(index)] for index in range(4)], orders["outer"]
        for count in (8, 16, 32):
            check()
            fitted, columns = fit_selected_rank(x, y, labels, folds, inner, outer, count, progress)
            selected = [channels[int(index)] for index in outer[:count]]
            expected_columns = [2*int(index)+band for index in outer[:count] for band in range(2)]
            if list(columns) != expected_columns:
                raise ArithmeticError("Selected decoder columns must be exactly two per physical electrode")
            trajectory = Trajectory(path, source, role, records, selected, check, exposure)
            trajectory.initialize_selected(x, y, channels, guide)
            for total in CHECKPOINTS:
                check()
                if total > 32:
                    fitted, _ = later_fit(trajectory, total, records, progress)
                models.append((seed, total, count, tuple(selected), fitted))
        for total in CHECKPOINTS:
            models.append((seed, total, 96, tuple(channels), shared[total]))
    if len(models) != 120:
        raise ArithmeticError("All120 model references/95 distinct models required before evaluation")
    return models, nulls, guide, selections


def score_target(models, nulls, x, y, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = {seed: [] for seed in BRANCHES}
    for seed, total, count, selected, fitted in models:
        columns = [2*positions[channel]+band for channel in selected for band in range(2)]
        prediction = predict_capacity(fitted, x[:, columns], labels)
        stats = fitted.statistics
        score = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
        denominator = float(np.sum(((y-stats.target_mean)/stats.target_sd)**2))
        correct = float(np.sum((((y-stats.target_mean)/stats.target_sd)-
                                ((prediction-stats.target_mean)/stats.target_sd))**2))
        wrong = (1-score["Q_wrong"])*denominator
        null = nulls[total]
        null_q = prediction_scores(null.direction_target_mean[labels], y, labels, null.target_mean, null.target_sd)["Q"]
        electrode_trials = 32*96+(total-32)*count
        rows[seed].append({"training_trials": total, "electrodes": count, "mc_seed": seed,
            "channels": list(selected), "rank": fitted.rank, "alpha": fitted.alpha,
            "candidate_count": fitted.candidate_count, "null_SSE": denominator,
            "standardized_SSE": correct, "standardized_wrong_SSE": wrong,
            "native_target_SSE": float(np.sum((y-prediction)**2)), **score,
            "direction_only_Q": null_q, "delta_Q_from_direction_only": score["Q"]-null_q,
            "LFP_electrode_trials": electrode_trials, "LFP_numeric_elements": electrode_trials*20,
            "target_numeric_elements": total*y.shape[1]*10, "initial_trials": 32, "initial_LFP_electrodes": 96,
            "input_coordinates_per_electrode": 2})
    branch_panels = {}
    for seed, fitted_rows in rows.items():
        fitted_rows.sort(key=lambda row: (row["training_trials"], row["electrodes"]))
        cells = [{key: value for key, value in row.items()
                  if key not in ("channels", "rank", "alpha", "candidate_count", "mc_seed")}
                 for row in fitted_rows]
        branch_panels[seed] = {"cells": cells, "fitted_models": fitted_rows,
            **{"mean_"+key: float(np.mean([cell[key] for cell in cells]))
               for key in ("Q", "Q_wrong", "pairing_increment", "delta_Q_from_direction_only")}}
    seeds = [{"mc_seed": seed, "panel": branch_panels[seed]} for seed in MC_SEEDS]
    control = mean_control_cells([entry["panel"] for entry in seeds])
    control["mean_delta_Q_from_direction_only"] = float(np.mean([cell["delta_Q_from_direction_only"] for cell in control["cells"]]))
    return {POLICY: branch_panels[None], CONTROL: control}, seeds


def attach_references(sessions, previous, lmp):
    if (previous['status'] != 'completed_one_initializer_target_label_control'
            or previous['held_session_accessed'] or previous['control'] != PREVIOUS_CONTROL
            or previous['MC_seeds'] != list(MC_SEEDS)
            or lmp['status'] != 'completed_two_LMP_correspondence_comparisons'
            or lmp['held_session_accessed'] or lmp['policy'] != LMP_POLICY
            or lmp['control'] != LMP_CONTROL or lmp['MC_seeds'] != list(MC_SEEDS)
            or lmp['recipe'] != 'outputs/lmp_only_correspondence_proposal.md'):
        raise ValueError('Complete original three-band and LMP-only references required')
    for packet in (previous, lmp):
        if len(packet['sessions']) != 8 or {s['filename'] for s in packet['sessions']} != set(DEVELOPMENT):
            raise ValueError('Exactly all eight original reference sessions required')
    old = {s['filename']: s for s in previous['sessions']}
    low = {s['filename']: s for s in lmp['sessions']}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    maximum, target_max = 0., 0.
    for session in sessions:
        reference, low_reference = old[session['filename']], low[session['filename']]
        new_maps = {(r['mc_seed'], r['view']): r['target_label_permutations']
                    for r in session['selection_records'] if r['mc_seed'] is not None}
        if len(new_maps) != 25:
            raise ValueError('Twenty-five complete control view maps required')
        for historical in (reference, low_reference):
            if (historical['animal'] != session['animal']
                    or historical['source_roles'] != session['source_roles']
                    or historical['target_unit_count'] != session['target_unit_count']):
                raise ValueError('Historical animal/trial/electrode/target-count support differs')
            maps = {(r['mc_seed'], r['view']): r['target_label_permutations']
                    for r in historical['selection_records'] if r['mc_seed'] is not None}
            if maps != new_maps:
                raise ArithmeticError('Original within-direction/initializer-view maps differ')
        reused = {'initial_marginal': reference['panels']['initial_marginal'],
                  PREVIOUS_CONTROL: reference['panels'][PREVIOUS_CONTROL],
                  LMP_POLICY: low_reference['panels'][LMP_POLICY],
                  LMP_CONTROL: low_reference['panels'][LMP_CONTROL]}
        historical_cells = {}
        for policy, panel in reused.items():
            index = {(c['training_trials'], c['electrodes']): c for c in panel['cells']}
            if len(panel['cells']) != 20 or set(index) != expected:
                raise ValueError('Every reused spectral panel needs exact twenty resource cells')
            base_fields = {'training_trials', 'electrodes', 'Q', 'Q_wrong', 'pairing_increment',
                           'direction_only_Q', 'delta_Q_from_direction_only', 'target_numeric_elements',
                           'LFP_electrode_trials', 'LFP_numeric_elements'}
            sse_fields = {'null_SSE', 'standardized_SSE', 'standardized_wrong_SSE'}
            for cell in index.values():
                if not base_fields <= cell.keys():
                    raise ValueError('Required historical scientific/exposure fields are missing')
                if policy == 'initial_marginal':
                    # The original acquired-only producer did not save these
                    # quantities. Permit exactly that legacy schema, not arbitrary
                    # missing keys in the later control or LMP packets.
                    if sse_fields & cell.keys() and not sse_fields <= cell.keys():
                        raise ValueError('Partial legacy SSE schema is not a qualified reference')
                elif not sse_fields <= cell.keys():
                    raise ValueError('Required control/LMP SSE denominator fields are missing')
            historical_cells[policy] = index
        # The LMP packet also retained the original three-band panels. Their
        # embedded scientific quantities must agree; added descriptive keys need not.
        for policy in ('initial_marginal', PREVIOUS_CONTROL):
            embedded = {(c['training_trials'], c['electrodes']): c
                        for c in low_reference['panels'][policy]['cells']}
            if len(low_reference['panels'][policy]['cells']) != 20 or set(embedded) != expected:
                raise ValueError('LMP packet must retain complete original three-band reference')
            for key in expected:
                for metric, value in historical_cells[policy][key].items():
                    if metric not in embedded[key] or embedded[key][metric] != value:
                        raise ValueError('Reused three-band scientific records disagree')
            if 'fitted_models' in reference['panels'][policy]:
                if (low_reference['panels'][policy].get('fitted_models')
                        != reference['panels'][policy]['fitted_models']):
                    raise ValueError('Reused three-band fitted-score records disagree')
        candidate_full = {r['training_trials']: r for r in session['panels'][POLICY]['fitted_models']
                          if r['electrodes'] == 96}
        if set(candidate_full) != set(CHECKPOINTS):
            raise ValueError('Every new full96 high-power checkpoint required')
        branches = [session['panels'][POLICY], *(seed['panel'] for seed in session['seed_panels'])]
        if len(branches) != 6 or [s['mc_seed'] for s in session['seed_panels']] != list(MC_SEEDS):
            raise ValueError('Candidate and all five ordinary control branches required')
        for panel in branches:
            if (len(panel['fitted_models']) != 20 or
                    {(r['training_trials'], r['electrodes']) for r in panel['fitted_models']} != expected):
                raise ValueError('Every new branch needs twenty exact fitted models')
            for row in panel['fitted_models']:
                n, e = row['training_trials'], row['electrodes']
                for policy, index in historical_cells.items():
                    before = index[n, e]
                    old_width = 1 if policy in (LMP_POLICY, LMP_CONTROL) else 3
                    if (row['target_numeric_elements'] != before['target_numeric_elements']
                            or row['LFP_electrode_trials'] != before['LFP_electrode_trials']
                            or row['LFP_numeric_elements']*old_width != before['LFP_numeric_elements']*2
                            or row['input_coordinates_per_electrode'] != 2):
                        raise ValueError('Target/physical-channel exposure or two-band bandwidth differs')
                    if policy != 'initial_marginal' or 'null_SSE' in before:
                        if not np.isclose(row['null_SSE'], before['null_SSE'], rtol=1e-10, atol=1e-10):
                            raise ArithmeticError('Representation-independent target denominator differs')
                    difference = abs(row['direction_only_Q']-before['direction_only_Q'])
                    if not np.isfinite(difference) or difference > 1e-10:
                        raise ArithmeticError('Representation-independent direction-null score differs')
                    target_max = max(target_max, difference)
                if e == 96:
                    own = candidate_full[n]
                    if any(row[key] != own[key] for key in ('channels', 'rank', 'alpha', 'candidate_count')):
                        raise ArithmeticError('Within-run full high-power capacity/support replay differs')
                    for key in ('Q', 'Q_wrong', 'pairing_increment'):
                        error = abs(row[key]-own[key])
                        if not np.isfinite(error) or error > 1e-10:
                            raise ArithmeticError('Within-run full high-power score replay differs')
                        maximum = max(maximum, error)
        for policy, panel in reused.items():
            session['panels'][policy] = copy.deepcopy(panel)
        indices = {policy: {(c['training_trials'], c['electrodes']): c
                           for c in session['panels'][policy]['cells']} for policy in ALL_POLICIES}
        if any(set(index) != expected or len(session['panels'][policy]['cells']) != 20
               for policy, index in indices.items()):
            raise ValueError('All twenty matching resource cells required')
        for policy in ALL_POLICIES:
            panel = session['panels'][policy]
            for cell in panel['cells']:
                key = cell['training_trials'], cell['electrodes']
                for other in ALL_POLICIES:
                    cell['delta_Q_from_'+other] = cell['Q']-indices[other][key]['Q']
            for other in ALL_POLICIES:
                panel['mean_delta_Q_from_'+other] = float(np.mean([c['delta_Q_from_'+other] for c in panel['cells']]))
            panel['mean_delta_Q_from_direction_only'] = float(np.mean([c['delta_Q_from_direction_only'] for c in panel['cells']]))
        own_contrast = session['panels'][POLICY]['mean_delta_Q_from_'+CONTROL]
        session['correspondence_spectral_contrasts'] = {
            'high_power_minus_three_band': own_contrast-session['panels']['initial_marginal']['mean_delta_Q_from_'+PREVIOUS_CONTROL],
            'high_power_minus_LMP': own_contrast-session['panels'][LMP_POLICY]['mean_delta_Q_from_'+LMP_CONTROL]}
    return maximum, target_max


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {session['filename'] for session in sessions} != set(DEVELOPMENT)
                     or any(sum(session['animal'] == animal for session in sessions) != 4
                            for animal in ('Chewie', 'Mihili'))):
        raise ValueError('All four original development sessions per animal required')
    keys = ('mean_Q', 'mean_Q_wrong', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
            *('mean_delta_Q_from_'+policy for policy in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{'animal': animal, **{key: float(np.mean([session['panels'][policy][key]
                    for session in sessions if session['animal'] == animal])) for key in keys}}
                   for animal in ('Chewie', 'Mihili')]
        panels.append({'policy': policy, 'animals': animals,
                       **{key: float(np.mean([animal[key] for animal in animals])) for key in keys}})
    return panels


def finalize_scientific_report(report, previous, lmp):
    """Use complete saved fit scores only; never acquire or refit neural data."""
    replay, target_replay = attach_references(report["sessions"], previous, lmp)
    report["within_run_full_high_power_score_replay_max_absolute_difference"] = replay
    report["historical_direction_null_Q_replay_max_absolute_difference"] = target_replay
    report["panels"] = aggregate(report["sessions"])
    report["fixed_prediction_session_omission"] = [
        {"excluded_session": name, "panels": aggregate([s for s in report["sessions"] if s["filename"] != name], False)}
        for name in DEVELOPMENT]
    true = next(panel for panel in report["panels"] if panel["policy"] == POLICY)
    report["prospective_directional_screens"] = {
        "true_high_power_marginal_positive_vs_control_in_both_animals":
            all(animal["mean_delta_Q_from_"+CONTROL] > 0 for animal in true["animals"]),
        "true_high_power_marginal_positive_vs_direction_null_in_both_animals":
            all(animal["mean_delta_Q_from_direction_only"] > 0 for animal in true["animals"]),
        "significance_or_promotion_claim": False}
    report["seed_summaries"] = []
    for seed in MC_SEEDS:
        animals = []
        for animal in ("Chewie", "Mihili"):
            subset = [s for s in report["sessions"] if s["animal"] == animal]
            seed_q = float(np.mean([next(entry["panel"]["mean_Q"] for entry in s["seed_panels"]
                                         if entry["mc_seed"] == seed) for s in subset]))
            true_q = float(np.mean([s["panels"][POLICY]["mean_Q"] for s in subset]))
            animals.append({"animal": animal, "mean_Q": seed_q, "true_minus_seed_Q": true_q-seed_q})
        report["seed_summaries"].append({"mc_seed": seed, "animals": animals,
            "mean_Q": float(np.mean([animal["mean_Q"] for animal in animals])),
            "true_minus_seed_Q": float(np.mean([animal["true_minus_seed_Q"] for animal in animals]))})
    report["correspondence_spectral_contrasts"] = {}
    for name in ("high_power_minus_three_band", "high_power_minus_LMP"):
        animals = [{"animal": animal, "contrast": float(np.mean([
            session["correspondence_spectral_contrasts"][name] for session in report["sessions"]
            if session["animal"] == animal]))} for animal in ("Chewie", "Mihili")]
        report["correspondence_spectral_contrasts"][name] = {
            "definition": name+" in true-minus-own-control Q; feature information and capacity differ",
            "animals": animals, "mean": float(np.mean([a["contrast"] for a in animals]))}
    report["reference_reuse"] = {"reference_report": report['reference_paths']['reference_report'], "lmp_reference": report['reference_paths']['lmp_reference'],
        "read_only_after_all_new_fits": True, "old_prediction_array_identity_claimed": False,
        "old_serialized_source_role_identity_verified": True,
        "old_native_unit_guide_identity_claimed": False, "old_model_refits_charged": 0}
    report.update(status="completed_two_high_power_correspondence_comparisons", stage="complete",
        valid_evaluations_charged=2, valid_evaluations_total=34, search_progress_resolution_Q=.001,
        scientific_scores_released=True)
    report["reference_reuse"]["legacy_initial_marginal_SSE_denominator_replay_claimed"] = False
    report["reference_reuse"]["legacy_initial_marginal_schema_note"] = (
        "Original acquired-only cells did not serialize null_SSE or standardized SSE; "
        "all recorded original fields replay exactly between embedded packets. "
        "Target denominators replay against original target-label control and both LMP panels.")
    return report
