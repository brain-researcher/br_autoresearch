#!/usr/bin/env python3
"""Explicit EP04 remaining-control slots; metadata preparation or one Slurm task.

Root supplies every numeric setting after inspecting first-pass costs. Preparation
reads only the authored plan and writes an explicit manifest/report-path lists.
Run executes one explicit participant/fold bundle, without retry. Collectors retain authority for
complete coverage and scientific/source identity; this is only dispatch plumbing.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

EPISODE = Path(__file__).resolve().parents[2]
PLAN = EPISODE / 'outputs/code/remaining_controls_plan.json'
MANIFEST_ROOT = EPISODE / 'outputs/code/remaining_controls_selected'
# Frozen run_training_context_rsa.PARTICIPANTS, inherited by both runners.
PARTICIPANTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def selected_path(value, root):
    path = (EPISODE / value).resolve()
    require(path.is_relative_to(EPISODE / root), 'Output outside the declared episode family')
    return path


def selected_partitions(value):
    require(isinstance(value, str), 'Explicit comma-separated partition list required')
    partitions = value.split(',')
    require(all(partition and not any(character.isspace() for character in partition)
                for partition in partitions) and len(set(partitions)) == len(partitions),
            'Selected partitions must be nonempty, distinct names without blank tokens')
    return partitions


def validate_slots(slots):
    """Reject concrete overlap; complete scientific coverage belongs to collectors."""
    seen_outputs, seen_encoding, intervals = set(), set(), {}
    require(slots and [s['slot'] for s in slots] == list(range(len(slots))), 'Slots must be contiguous')
    require([(slot['participant'], slot['fold']) for slot in slots]
            == [(person, fold) for person in PARTICIPANTS for fold in range(5)], 'One explicit bundle for each frozen cell')
    for slot in slots:
        require(slot['actions'] and all((task['participant'], task['fold']) == (slot['participant'], slot['fold'])
                                       for task in slot['actions']), 'Bundle may contain only its declared cell')
        require(any(task['kind'] == 'null' for task in slot['actions']), 'Do not submit a short encoding-only task')
    for task in (task for slot in slots for task in slot['actions']):
        cell = (task['participant'], task['fold'])
        require(cell[0] in PARTICIPANTS and type(cell[1]) is int and cell[1] in range(5), 'Unknown cell')
        kind = task['kind']
        require(kind in ('encoding', 'null'), 'Unknown control task')
        family = 'identical_fold_encoding' if kind == 'encoding' else 'registered_nulls'
        output = selected_path(task['output_dir'], f'outputs/runs/{family}/20261005-remaining01')
        require(output not in seen_outputs, 'Duplicate output path')
        seen_outputs.add(output)
        require(type(task['max_seconds']) is int and 0 < task['max_seconds'] <=
                (1200 if kind == 'encoding' else 14400), 'Process cap exceeds existing runner limit')
        if kind == 'encoding':
            require(cell != ('sub-01', 0) and cell not in seen_encoding, 'Retained/duplicate encoding cell')
            seen_encoding.add(cell)
            continue
        nonempty = False
        for field, limit, retained_stop in (('permutation_range', 10000, 256), ('geometry_range', 1000, 8)):
            interval = task[field]
            require(isinstance(interval, list) and len(interval) == 2 and
                    all(type(v) is int for v in interval) and 0 <= interval[0] <= interval[1] <= limit,
                    'Invalid half-open draw range')
            start, stop = interval
            if start == stop:
                continue
            require(cell != ('sub-01', 0) or start >= retained_stop, 'Overlaps retained null prefix')
            nonempty = True
            intervals.setdefault((cell, field), []).append((start, stop))
        require(nonempty, 'Empty null task')
    for ranges in intervals.values():
        ranges.sort()
        require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), 'Overlapping null slots')


def bundle_seconds(slot):
    # Each existing process has internal +10s TERM deadline and +10s kill grace.
    return 50 + sum(task['max_seconds']+20 for task in slot['actions'])


def prepare(args):
    plan = read(PLAN)
    selected_partitions(args.partition)
    require(plan['status'] == 'unselected_awaiting_first_pass_costs' and not plan['slots'], 'Use authored unselected plan')
    require([(cell['participant'], cell['fold']) for cell in plan['remaining_cells']]
            == [(person, fold) for person in PARTICIPANTS for fold in range(5)], 'Exact frozen 25-cell plan required')
    for cell in plan['remaining_cells']:
        pilot = (cell['participant'], cell['fold']) == ('sub-01', 0)
        require(cell['permutation_range'] == [256 if pilot else 0, 10000]
                and cell['geometry_range'] == [8 if pilot else 0, 1000]
                and (cell['encoding_output_dir'] is None) == pilot, 'Preserve exact remaining work and pilot cell')
    require(0 < args.permutation_chunk <= 10000 and 0 < args.geometry_chunk <= 1000, 'Select positive measured chunk lengths')
    require(0 < args.encoding_seconds <= 1200 and 0 < args.null_seconds <= 14400, 'Existing runner caps apply')
    require(0 < args.memory_mib <= 8000 and 0 < args.concurrency <= 48, 'One CPU per slot; at most 48 concurrent cores')
    require(math.isfinite(args.remaining_cpu_hours) and 0 < args.remaining_cpu_hours <= 2500, 'Supply coordinator remaining budget')
    slots = []
    for cell in plan['remaining_cells']:
        actions = []
        if cell['encoding_output_dir'] is not None:
            actions.append(dict(kind='encoding', participant=cell['participant'], fold=cell['fold'],
                                output_dir=cell['encoding_output_dir'], max_seconds=args.encoding_seconds))
        p, p_end = cell['permutation_range']
        g, g_end = cell['geometry_range']
        while p < p_end or g < g_end:
            pn, gn = min(p+args.permutation_chunk, p_end), min(g+args.geometry_chunk, g_end)
            name = f"{cell['participant']}-fold-{cell['fold']}-p{p:05d}-{pn:05d}-g{g:04d}-{gn:04d}"
            actions.append(dict(kind='null', participant=cell['participant'], fold=cell['fold'],
                                permutation_range=[p, pn], geometry_range=[g, gn],
                                output_dir=f"{plan['null_output_root']}/{name}", max_seconds=args.null_seconds))
            p, g = pn, gn
        slots.append(dict(slot=len(slots), participant=cell['participant'], fold=cell['fold'], actions=actions))
    validate_slots(slots)
    require(args.wall_seconds >= max(bundle_seconds(slot) for slot in slots), 'Allow every bundled action and termination headroom')
    reservation = len(slots)*args.wall_seconds/3600
    require(reservation <= args.remaining_cpu_hours, 'Worst-case allocated CPU reservation exceeds supplied remaining budget')
    output = (EPISODE / args.output_dir).resolve()
    require(output.parent == MANIFEST_ROOT, 'Use one fresh named directory in outputs/code/remaining_controls_selected')
    MANIFEST_ROOT.mkdir(exist_ok=True)
    output.mkdir()  # Fresh manifest family; no overwrite/replay.
    manifest = {**plan, 'status': 'coordinator_selected', 'dispatch': 'one_participant_fold_bundle_per_slot', 'slots': slots,
                'selected_resources': {'cpus_per_task': 1, 'memory_mib': args.memory_mib,
                    'wall_seconds': args.wall_seconds, 'array_concurrency': args.concurrency,
                    'partition': args.partition, 'reserved_cpu_hours': reservation,
                    'coordinator_remaining_cpu_hours': args.remaining_cpu_hours}}
    write(output/'manifest.json', manifest)
    for kind, pilot, report_name in (
            ('encoding', plan['retained_encoding_report'], 'identical_fold_encoding.json'),
            ('null', plan['retained_null_report'], 'registered_null_unit.json')):
        reports = [str(EPISODE/pilot)] + [str(EPISODE/t['output_dir']/report_name)
                                        for slot in slots for t in slot['actions'] if t['kind'] == kind]
        write(output/f'{kind}_reports.json', reports)
    # Resource/partition verification and shared concurrency accounting remain root's responsibility.
    command = ['sbatch', '--partition', args.partition, '--mem', f'{args.memory_mib}M',
               '--time', f'{args.wall_seconds//3600:02d}:{args.wall_seconds//60%60:02d}:{args.wall_seconds%60:02d}',
               '--array', f'0-{len(slots)-1}%{args.concurrency}',
               '--chdir', str(EPISODE), '--output', str(output/'%A_%a.out'),
               '--error', str(output/'%A_%a.err'),
               str(EPISODE/'outputs/code/remaining_controls_array.sbatch'), str(output/'manifest.json')]
    print(json.dumps({'manifest': str(output/'manifest.json'), 'slots': len(slots),
                      'reserved_cpu_hours': reservation, 'submission_not_executed': shlex.join(command)}, indent=2))


def completed_encoding_not_evaluable(task, status):
    """Exit 2 also represents a fully recorded scientific non-evaluability result."""
    if task['kind'] != 'encoding' or status != 2:
        return False
    try:
        output = selected_path(task['output_dir'], 'outputs/runs/identical_fold_encoding/20261005-remaining01')
        report = read(output/'identical_fold_encoding.json')
        return (isinstance(report, dict) and report.get('participant') == task['participant']
                and type(report.get('held_fold')) is int and report['held_fold'] == task['fold']
                and report.get('diagnostic') == 'identical_fold_encoding'
                and report.get('all_16_rows_recorded') is True
                and report.get('status') == 'encoding_diagnostic_not_evaluable'
                and report.get('complete_encoding_diagnostic') is False and report.get('errors') == [])
    except (OSError, ValueError):
        return False


def run(args):
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in socket.gethostname(), 'Compute allocation required')
    require(Path.cwd().resolve() == EPISODE, 'Use EP04 cwd')
    manifest_path = (EPISODE / args.manifest).resolve()
    require(manifest_path.name == 'manifest.json' and manifest_path.parent.parent == MANIFEST_ROOT,
            'Read only an explicit prepared remaining_controls_selected/*/manifest.json')
    manifest = read(manifest_path)
    require(manifest['status'] == 'coordinator_selected'
            and manifest.get('dispatch') == 'one_participant_fold_bundle_per_slot', 'Require selected bundled manifest')
    slots, resources = manifest['slots'], manifest['selected_resources']
    allowed_partitions = selected_partitions(resources['partition'])
    validate_slots(slots)
    require(os.environ.get('SLURM_CPUS_PER_TASK') == '1' and os.environ.get('SLURM_JOB_CPUS_PER_NODE') == '1'
            and os.environ.get('SLURM_NNODES') == '1' and all(os.environ.get(k) == '1' for k in THREADS),
            'Exactly one allocated CPU/node and six thread limits=1 required')
    require(int(os.environ['SLURM_MEM_PER_NODE']) == resources['memory_mib']
            and os.environ['SLURM_JOB_PARTITION'] in allowed_partitions, 'Allocation differs from manifest')
    require(int(os.environ['SLURM_ARRAY_TASK_COUNT']) == len(slots) and 0 < resources['array_concurrency'] <= 48,
            'Array differs from explicit manifest')
    slot = int(os.environ['SLURM_ARRAY_TASK_ID'])
    require(0 <= slot < len(slots), 'Unknown array slot')
    bundle = slots[slot]
    require(resources['wall_seconds'] >= bundle_seconds(bundle), 'Insufficient selected bundle headroom')
    # Reserved pilot cells are never rerun or replaced by this dispatcher.
    plan = read(PLAN)
    for key in ('observed_report', 'low_level_dir', 'retained_encoding_report', 'retained_null_report'):
        require(manifest[key] == plan[key], 'Fixed observed/pilot binding changed')
    pilot_encoding = read(EPISODE/plan['retained_encoding_report'])
    pilot_null = read(EPISODE/plan['retained_null_report'])
    require(pilot_encoding['participant'] == 'sub-01' and pilot_encoding['held_fold'] == 0
            and pilot_encoding.get('diagnostic') == 'identical_fold_encoding'
            and pilot_encoding.get('status') in ('encoding_diagnostic_complete', 'encoding_diagnostic_not_evaluable')
            and pilot_encoding.get('all_16_rows_recorded') is True, 'Retained encoding pilot incomplete')
    require(pilot_null['participant'] == 'sub-01' and pilot_null['held_fold'] == 0
            and pilot_null.get('mode') == 'registered_null_participant_fold_unit'
            and pilot_null.get('status') == 'completed_selected_null_ranges'
            and pilot_null.get('complete_selected_ranges') is True
            and pilot_null['permutation_range'] == [0, 256] and pilot_null['geometry_range'] == [0, 8],
            'Retained null prefix incomplete')
    require(all(not (EPISODE/task['output_dir']).exists() for task in bundle['actions']),
            'Bundle output already exists; no automatic retry')
    for action, task in enumerate(bundle['actions']):
        runner = 'run_identical_fold_encoding.py' if task['kind'] == 'encoding' else 'run_registered_null_unit.py'
        command = ['timeout', '--signal=TERM', '--kill-after=10s', f"{task['max_seconds']+10}s",
                   sys.executable, '-u', '-B', str(EPISODE/'outputs/code'/runner),
                   '--participant', task['participant'], '--fold', str(task['fold']),
                   '--observed-report', manifest['observed_report'], '--low-level-dir', manifest['low_level_dir'],
                   '--output-dir', task['output_dir'], '--max-seconds', str(task['max_seconds'])]
        if task['kind'] == 'null':
            for field, flag in (('permutation_range', '--permutation-range'), ('geometry_range', '--geometry-range')):
                command += [flag, ':'.join(map(str, task[field]))]
        print(json.dumps({'slot': slot, 'action': action, 'task': task, 'command': command}), flush=True)
        # Existing runner owns source checks, partial results and the original 120h deadline.
        status = subprocess.call(command)
        print(json.dumps({'slot': slot, 'action': action, 'exit_status': status}), flush=True)
        if status != 0:
            if completed_encoding_not_evaluable(task, status):
                print(json.dumps({'slot': slot, 'action': action,
                                  'encoding_status': 'encoding_diagnostic_not_evaluable',
                                  'all_rows_recorded': True, 'continuing_independent_null_actions': True}), flush=True)
                continue  # Preserve the non-evaluable report; collectors/screen still enforce its consequences.
            return status  # Preserve completed actions; root diagnoses before any further execution.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--output-dir', type=Path, required=True)
    for name in ('permutation-chunk', 'geometry-chunk', 'encoding-seconds', 'null-seconds',
                 'memory-mib', 'wall-seconds', 'concurrency'):
        prepare_parser.add_argument('--'+name, type=int, required=True)
    prepare_parser.add_argument('--remaining-cpu-hours', type=float, required=True)
    prepare_parser.add_argument('--partition', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args)
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
