"""Continue a saved Stage-1 reference frame without changing reference physics.

Restore the last complete action checkpoint, replay every saved split after it
with exact nonenergy observable/diagnostic comparisons and at most eight ULP
for separately reduced nonzero physical energies, then continue the same cycle-0
B2/2001 offset fixture. Save compatible interval/checkpoint archives at every
dwell. The original run and all reference/fast numerical modules are read-only.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np
import fast_solver_reference as reference

ROOT = reference.EPISODE_ROOT / 'outputs/runs/fast_forward_stage1/solver'
CONVERSION_KEYS = ('conv_start', 'conv_end', 'conv_adc', 'conv_marker', 'pixel_conv', 'pixel_site')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_archives(out, records, conversions, checkpoints):
    # Intervals are written first. A termination between renames leaves the
    # older state checkpoint supported by the newer interval prefix.
    for name, values in (
        ('reference_intervals.npz', {**records, **conversions, 'work_fields': np.array(reference.WORK_FIELDS)}),
        ('reference_checkpoints.npz', checkpoints),
    ):
        path = out / name
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def restore_checkpoint(array, state, field):
    """The reference has no public restart API; restore its persistent state only.

    SparsePassiveRCNetwork stores volts; NeurocamCPEArray stores the previous
    field for boundary charge/work. Its fixed C factor is state-independent.
    Every advance rebuilds its own interval integrator, so no timestep history
    is omitted. Noise binding is established before setting _advanced.
    """
    if (state.shape != (69848,) or field.shape != (4096,)
            or not np.isfinite(state).all() or not np.isfinite(field).all()):
        raise ValueError('Require all finite reference states and prior-field values')
    if array._network._pending_preview is not None:
        raise ValueError('Fresh reference unexpectedly has a pending preview')
    array._network._volts = state.copy()
    array._last_field = field.copy()
    array._advanced = True
    np.testing.assert_array_equal(np.asarray(array.volts), state)
    np.testing.assert_array_equal(np.asarray(array.previous_field_volts), field)


def interval_values(result, held, action_index, hold_index, start, end, elapsed):
    kernel = result.kernel_interval
    end_volts, mean = np.asarray(kernel.observation.end_volts), np.asarray(kernel.observation.mean_volts)
    return {
        't_start': start, 't_end': end, 'action_index': action_index, 'hold_index': hold_index,
        'field_key': held.field_key, 'gate_row': held.gate_row,
        'hold_mean': [v.mean_volts for v in result.daq_hold_observations],
        'hold_end': [v.end_volts for v in result.daq_hold_observations],
        'source_mean': mean[4096:4160], 'source_end': end_volts[4096:4160],
        'buffer_end': end_volts[69696:69760], 'filter_end': end_volts[69760:69824],
        'buffer_mean': mean[69696:69760],
        'work': [getattr(result, name) for name in reference.WORK_FIELDS],
        'energy': (result.physical_energy_before_boundary_joule,
                   result.physical_energy_after_boundary_joule, result.physical_energy_end_joule),
        'subintervals': kernel.subintervals, 'partition_error': kernel.maximum_scaled_partition_error,
        'endpoint_error': kernel.estimated_endpoint_error_volts, 'mean_error': kernel.estimated_mean_error_volts,
        'energy_residual': kernel.energy_balance_residual_joule,
        'numerical_dissipation': kernel.numerical_dissipation_joule,
        'operator_products': kernel.operator_products, 'elapsed': elapsed,
    }


def energy_replay_check(previous, actual, index):
    """Bound only the three separately reduced physical energies, without a floor.

    These nonnegative quadratic reductions occur after the reference integrator
    computes its trajectory and work. This check does not alter integration,
    partition refinement, physical-work tolerances or any nonenergy comparison.
    """
    previous, actual = np.asarray(previous, dtype=float), np.asarray(actual, dtype=float)
    if previous.shape != (3,) or actual.shape != (3,) or not np.isfinite(previous).all() or not np.isfinite(actual).all():
        raise ValueError('Energy replay requires three finite original and replayed values')
    difference = np.abs(actual-previous)
    nonzero = previous != 0
    spacing = np.spacing(np.abs(previous))
    ratios = [float(difference[i]/spacing[i]) if nonzero[i] else (0.0 if difference[i] == 0 else None)
              for i in range(3)]
    passed = bool(np.all(difference[nonzero] <= 8*spacing[nonzero])
                  and np.all(difference[~nonzero] == 0))
    return {'interval': index, 'exact': bool(np.array_equal(previous, actual)), 'passed': passed,
            'maximum_allowed_ULP': 8, 'zero_reference_requires_exact': True,
            'original': previous.tolist(), 'replayed': actual.tolist(),
            'absolute_difference': difference.tolist(), 'reference_ULP': spacing.tolist(),
            'ULP_counts': ratios,
            'original_hex': [float(v).hex() for v in previous],
            'replayed_hex': [float(v).hex() for v in actual]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=float, required=True)
    args = parser.parse_args()
    host, job = socket.gethostname(), os.environ.get('SLURM_JOB_ID', '')
    if '-ln' in host or not job:
        raise SystemExit('An actual Slurm compute shell is required')
    source, out = args.source.resolve(), args.run_dir.resolve()
    if (not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT)
            or not source.is_dir() or not out.is_dir()
            or any((out / name).exists() for name in ('settings.json', 'reference_intervals.npz', 'reference_report.json'))):
        raise SystemExit('Use an existing source and a fresh existing output directory inside the solver root')
    if not 0 < args.deadline_seconds <= 25920:
        raise SystemExit('Internal bound must be positive and at most 7.2 hours')
    started = time.monotonic()
    interrupted = {'signal': None}
    def on_signal(signum, frame):
        interrupted['signal'] = signum
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    def check():
        if interrupted['signal'] is not None or time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError('Continuation deadline or signal; preserve completed checkpoints')

    report = {'status': 'failed', 'source': str(source), 'host': host, 'slurm_job_id': job,
              'slots_requested': 1600, 'deadline_seconds': args.deadline_seconds,
              'reference_modules_modified': False, 'overlap_exact': False, 'overlap_valid': False,
              'energy_replay_checks': [], 'checkpoint_state_checks': []}
    records = conversions = checkpoints = None
    phase, completed = 'source', 0
    try:
        from analog_conditioning import AnalogConditioning, SourceConditioning
        from conditioned_array_solver import ConditionedSolverControls
        from conditional_acquisition_recipe import build_conditional_frontend, build_conditional_schedule, conditional_noise_cell
        from conditional_forward_recipe import (CONDITIONING, REFERENCE_ID, build_conditional_circuit_parameters,
            build_conditional_noise, expected_state_ids, metadata)
        from cpe_array import NeurocamCPEArray
        from development_simulation import load_saved_development_world
        from raw_input_hold import compile_midpoint_hold
        from raw_world_time import RawWorldTime
        from surface_to_contact import build_contact_factors, registered_pad_geometry

        settings = json.loads((source / 'settings.json').read_text())
        if (settings['slots_requested'] != 1600 or settings['tolerance'] != 'standard'
                or settings['world_path'] != str(reference.WORLD_PATH)
                or settings['noise_path'] != str(reference.NOISE_PATH)
                or settings['electrical'] != metadata()
                or settings['field_hold_epsilon_volts'] != 1e-8
                or settings['noise_current_hold_tolerance_amp'] != 1e-12
                or settings['noise_voltage_hold_tolerance_volts'] != 1e-8):
            raise ValueError('Source is not the selected standard cycle-0 B2/2001 recipe')
        offsets = settings['offsets_volts']
        expected_offsets = json.loads((ROOT / 'offsets_fixture.json').read_text())
        if offsets != expected_offsets or len(offsets) != 64:
            raise ValueError('Require the unchanged selected 64-channel offset fixture')
        controls = ConditionedSolverControls(**settings['solver_controls'])
        expected_controls = ConditionedSolverControls(relative_tolerance=1e-5, voltage_atol=1e-8,
            charge_atol_coulomb=1e-20, max_refinements=12, max_operator_products=200000,
            linear_relative_tolerance=1e-9, linear_charge_atol_coulomb=1e-22,
            energy_atol_joule=1e-18, max_newton_iterations=12, max_linear_iterations=10000,
            gmres_restart=40, max_workspace_bytes=512*1024**2, integration_method='radau_iia_2',
            radau_preconditioner='contact_block', radau_linear_solver='triangular_contact_schur')
        if asdict(controls) != asdict(expected_controls):
            raise ValueError('Source numerical controls differ from the unchanged offset reference')
        with np.load(source / 'reference_intervals.npz', allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        with np.load(source / 'reference_checkpoints.npz', allow_pickle=False) as archive:
            original_checkpoints = {key: archive[key] for key in archive.files}
        finite = np.flatnonzero(np.isfinite(original['t_end']))
        if not np.array_equal(finite, np.arange(len(finite))) or not len(finite):
            raise ValueError('Source intervals must be a nonempty contiguous prefix')
        frontier = len(finite)
        supported = np.flatnonzero(original_checkpoints['times'] <= original['t_end'][frontier-1])
        if not len(supported):
            raise ValueError('No saved state checkpoint supported by the interval prefix')
        selected = int(supported[-1])
        resume_slot = int(original_checkpoints['slots'][selected])
        resume_time = float(original_checkpoints['times'][selected])
        if resume_slot >= 1600:
            raise ValueError('The source already contains a complete frame; do not rerun')
        cut = int(np.searchsorted(original['t_end'][:frontier], resume_time, side='right'))
        if cut == 0 or original['t_end'][cut-1] != resume_time:
            raise ValueError('Checkpoint is not exactly a committed split endpoint')
        state, prior_field = original_checkpoints['states'][selected], original_checkpoints['fields'][selected]
        report.update(resume_slot=resume_slot, resume_time=resume_time, reused_intervals=cut,
                      source_completed_intervals=frontier, overlap_intervals=frontier-cut)
        atomic_json(out/'settings.json', {**settings, **report,
            'checkpoint_slots': sorted(set(settings['checkpoint_slots']) | set(range(25, 1601, 25))),
            'restart_state': 'all_69848_voltages_plus_4096_previous_field_values',
            'overlap_check': 'exact_nonenergy_observables_works_diagnostics_and_full_saved_states;physical_energy_maximum_8_ULP_no_floor;elapsed_excluded'})

        phase = 'world_and_circuit'
        world = load_saved_development_world(reference.WORLD_PATH, condition_id='B2', replicate_seed=2001,
            max_elements=2000000, max_array_bytes=64*1024**2, max_metadata_bytes=1024**2, check=check)
        if world.nuisance_arrays:
            raise ValueError('B2 must have no nuisance arrays')
        geometry = registered_pad_geometry('G0', short_axis='x', max_elements=2000000, check=check)
        contacts = build_contact_factors(world, geometry, max_elements=2000000, check=check)
        raw_time = RawWorldTime(world, max_elements=2000000, check=check)
        frontend = build_conditional_frontend()
        parameters = build_conditional_circuit_parameters(daq_frontend=frontend, check=check)
        parameters['analog_conditioning'] = AnalogConditioning(
            channels=tuple(SourceConditioning(**dict(CONDITIONING, offset_volts=float(v))) for v in offsets),
            source_load_role='additional_fixed_shunt_excludes_conditioning',
            source_capacitance_role='additional_fixed_capacitance_excludes_conditioning', voltage_reference_id=REFERENCE_ID)
        array = NeurocamCPEArray(**parameters)
        if array.state_ids != expected_state_ids():
            raise ValueError('Reference state ordering differs')
        schedule = build_conditional_schedule(frontend, state_ids=array.state_ids, check=check)
        actions = tuple(schedule.physical_actions_for_cycle(0))
        readout = schedule.readout_window(0)
        noise = build_conditional_noise(array=array, check=check, retained_draw_path=None,
            current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
            max_evaluations=200000000, max_term_evaluations=200000000000000,
            reuse_draw_path=reference.NOISE_PATH, cubic_cell_for_key=conditional_noise_cell)
        restore_checkpoint(array, state, prior_field)
        report['checkpoint_state_checks'].append({'slot': resume_slot, 'time': resume_time,
                                                  'state_exact': True, 'previous_field_exact': True})

        phase = 'holds'
        plan = compile_midpoint_hold(raw_time, contacts, actions, epsilon_input_volts=1e-8,
            max_intervals=40000000, max_work=100000000, check=check, noise_hold_bound=noise.hold_bound)
        boundaries = sorted({p for c in readout.conversions for p in (c.aperture_start, c.aperture_end)})
        boundary_array = np.asarray(boundaries)
        splits, position = [], 0
        for hold_index, held in enumerate(plan.intervals):
            while not actions[position].start <= held.start < actions[position].end:
                position += 1
            lo = np.searchsorted(boundary_array, held.start, side='right')
            hi = np.searchsorted(boundary_array, held.end, side='left')
            points = [held.start, *boundaries[lo:hi], held.end]
            splits.extend((position, hold_index, a, b) for a, b in zip(points, points[1:]))
        if len(splits) != len(original['t_end']) or len(splits) != 6400:
            raise ValueError('Rebuilt full frame must retain all 6400 original splits')
        np.testing.assert_array_equal(original['t_start'][:frontier], [v[2] for v in splits[:frontier]])
        np.testing.assert_array_equal(original['t_end'][:frontier], [v[3] for v in splits[:frontier]])
        np.testing.assert_array_equal(original['action_index'][:frontier], [v[0] for v in splits[:frontier]])
        np.testing.assert_array_equal(original['hold_index'][:frontier], [v[1] for v in splits[:frontier]])
        np.testing.assert_array_equal(original['work_fields'], np.array(reference.WORK_FIELDS))
        if splits[cut][0] != resume_slot or splits[cut][2] != resume_time:
            raise ValueError('Restart must begin at the next complete action boundary')
        conversions = {key: original[key].copy() for key in CONVERSION_KEYS}
        np.testing.assert_array_equal(conversions['conv_start'], [c.aperture_start for c in readout.conversions])
        np.testing.assert_array_equal(conversions['conv_end'], [c.aperture_end for c in readout.conversions])
        np.testing.assert_array_equal(conversions['conv_adc'], [int(c.adc_id.rsplit('_', 1)[1]) for c in readout.conversions])
        records = {key: value.copy() for key, value in original.items() if key not in (*CONVERSION_KEYS, 'work_fields')}
        for value in records.values():
            value[cut:] = np.nan if np.issubdtype(value.dtype, np.floating) else 0
        checkpoints = {key: value[:selected+1].copy() for key, value in original_checkpoints.items()}
        completed = cut
        save_archives(out, records, conversions, checkpoints)

        phase = 'overlap_then_continuation'
        polynomial = field = current_action = current_hold = None
        overlap_energy_exact = True
        for index in range(cut, len(splits)):
            check()
            action_index, hold_index, start, end = splits[index]
            held = plan.intervals[hold_index]
            if current_action != action_index:
                polynomial = noise(held.field_key)
                array.validate_noise_interval(polynomial, daq_selected_sources=held.daq_selected_sources)
                if (polynomial.support_start_seconds != actions[action_index].start
                        or polynomial.support_end_seconds != actions[action_index].end):
                    raise ValueError('Noise cubic does not cover its original action')
                current_action = action_index
            if current_hold != hold_index:
                field = plan.field_provider(held.field_key)
                current_hold = hold_index
            charge = np.zeros(len(array.state_ids))
            if start == held.start:
                for node, value in held.injected_charge_coulomb:
                    charge[node] = value
            clock = time.monotonic()
            result = array.advance_row(gate_row=held.gate_row, duration_seconds=end-start, field_volts=field,
                voltage_reference_id=held.voltage_reference_id, observed_sources=(0,),
                injected_charge_coulomb=charge, controls=controls, daq_selected_sources=held.daq_selected_sources,
                check=check, noise_sample=polynomial, interval_start_seconds=start)
            values = interval_values(result, held, action_index, hold_index, start, end, time.monotonic()-clock)
            if index < frontier:
                energy_check = energy_replay_check(original['energy'][index], values['energy'], index)
                report['energy_replay_checks'].append(energy_check)
                overlap_energy_exact = overlap_energy_exact and energy_check['exact']
                # Save every comparison immediately, including a failed energy
                # bound, so a later hard stop cannot erase the seam evidence.
                atomic_json(out/'overlap_replay.json', {
                    'energy_replay_checks': report['energy_replay_checks'],
                    'checkpoint_state_checks': report['checkpoint_state_checks'],
                    'overlap_complete': False, 'overlap_exact': False, 'overlap_valid': False})
                mismatches = [key for key, value in values.items()
                              if key not in ('elapsed', 'energy')
                              and not np.array_equal(np.asarray(value), original[key][index])]
                if not energy_check['passed']:
                    mismatches.append('energy_exceeds_8_ULP_or_nonexact_zero')
                if mismatches:
                    diagnostics = {}
                    for key, value in values.items():
                        if key == 'elapsed':
                            continue
                        actual, previous = np.asarray(value), np.asarray(original[key][index])
                        entry = {'exact': bool(np.array_equal(actual, previous)),
                                 'replayed_shape': list(actual.shape), 'original_shape': list(previous.shape)}
                        if actual.shape == previous.shape:
                            absolute = np.abs(actual.astype(float)-previous.astype(float))
                            nonzero = previous != 0
                            entry['max_abs'] = float(np.max(absolute))
                            entry['max_rel_over_nonzero_reference'] = (
                                float(np.max(absolute[nonzero]/np.abs(previous[nonzero]))) if np.any(nonzero) else None)
                        if key == 'energy':
                            entry.update(original=previous.tolist(), replayed=actual.tolist(),
                                original_hex=[float(v).hex() for v in previous],
                                replayed_hex=[float(v).hex() for v in actual])
                        diagnostics[key] = entry
                    report['overlap_mismatch'] = {'interval': index, 'fields': mismatches,
                                                  'diagnostics': diagnostics}
                    raise ValueError(f'Restored overlap differs from original at interval {index}, fields {mismatches}')
            saved_at_end = np.flatnonzero(original_checkpoints['times'] == end)
            for saved_index in saved_at_end:
                exact_state = bool(np.array_equal(np.asarray(array.volts), original_checkpoints['states'][saved_index]))
                exact_field = bool(np.array_equal(np.asarray(array.previous_field_volts), original_checkpoints['fields'][saved_index]))
                state_check = {'slot': int(original_checkpoints['slots'][saved_index]), 'time': end,
                               'state_exact': exact_state, 'previous_field_exact': exact_field}
                report['checkpoint_state_checks'].append(state_check)
                if not exact_state or not exact_field:
                    report['checkpoint_state_failure'] = state_check
                    raise ValueError(f'Full saved checkpoint state differs at time {end}')
            if index + 1 >= frontier and not report['overlap_valid']:
                report['overlap_exact'] = overlap_energy_exact
                report['overlap_valid'] = True
                atomic_json(out/'overlap_replay.json', {
                    'energy_replay_checks': report['energy_replay_checks'],
                    'checkpoint_state_checks': report['checkpoint_state_checks'],
                    'overlap_complete': True, 'overlap_exact': report['overlap_exact'], 'overlap_valid': True})
            for key, value in values.items():
                records[key][index] = value
            completed = index+1
            last_of_action = index+1 == len(splits) or splits[index+1][0] != action_index
            done = action_index+1
            if last_of_action and done % 25 == 0:
                checkpoints['times'] = np.append(checkpoints['times'], end)
                checkpoints['slots'] = np.append(checkpoints['slots'], done)
                checkpoints['states'] = np.vstack((checkpoints['states'], np.asarray(array.volts)))
                checkpoints['fields'] = np.vstack((checkpoints['fields'], np.asarray(array.previous_field_volts)))
                save_archives(out, records, conversions, checkpoints)
                print(json.dumps({'event': 'checkpoint', 'actions': done, 'completed_intervals': completed,
                                  'elapsed_seconds': time.monotonic()-started}), flush=True)
        report['status'] = 'complete'
        phase = 'complete'
    except BaseException as exc:
        report['failure'] = {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)[:2000]}
    finally:
        if records is not None and checkpoints is not None:
            try:
                save_archives(out, records, conversions, checkpoints)
            except BaseException as exc:
                report['save_failure'] = str(exc)[:500]
                report['status'] = 'failed'
        report.update(completed_intervals=completed, elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if records is not None and completed:
            report['physical_frontier_seconds'] = float(records['t_end'][completed-1])
            report['solver_seconds'] = float(np.nansum(records['elapsed']))
            report['new_solver_seconds'] = float(np.nansum(records['elapsed'][report['reused_intervals']:]))
        atomic_json(out/'reference_report.json', report)
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
