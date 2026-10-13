#!/usr/bin/env python3
"""Freeze EP04 unique-image metadata folds; never open neural/image/feature values.

The only source reads are the exact stimulus CSV, selected-object receipt and
150 named trial TSVs. A compute-only acquisition pins GitHub main once, then
retrieves four small grouping/correction JSONs and their README at that commit.
Acquired provider metadata is preserved; derived row-level exports contain
only regular participant-unique images. Other roles appear only as counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from urllib.request import Request, urlopen

import numpy as np

PARTICIPANTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
STRATA = ('laion', 'things', 'thingsplus')
CELLS = tuple((p, s) for p in PARTICIPANTS for s in STRATA)
SOURCE_NAMES = {'LAION': 'laion', 'THINGS': 'things',
                'THINGS+': 'thingsplus', 'THINGSplus': 'thingsplus'}
FILES = ('README.md', 'data/pooled_image_groups.json',
         'data/pooled_duplicate_groups.json', 'data/duplicate_groups.json',
         'data/random_fold_changes.json')
CUTOFFS = {'clip': .25, 'dreamsim': .30, 'dinov2': .25, 'sscd': .60}
UNIQUE_REPEAT_DESIGN = 'https://laion-fmri.hebartlab.com/experimental_design.html#main-experiment'
TRIAL_AXIS_SOURCE = 'https://laion-fmri.hebartlab.com/glmsingle_betas.html#relation-to-stimuli'


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use the coordinator-owned verified Slurm allocation')


def get_small(url, maximum):
    request = Request(url, headers={'User-Agent': 'EP04-metadata-preparation',
                                   'Accept': 'application/vnd.github+json'})
    with urlopen(request, timeout=25) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Selected provider metadata exceeds bounded acquisition size')
    return data


def acquire_provider(destination, revision=None):
    destination.mkdir()
    if revision is None:
        api = 'https://api.github.com/repos/ViCCo-Group/LAION-fMRI/git/ref/heads/main'
        revision = json.loads(get_small(api, 100_000))['object']['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Provider commit response is not an immutable commit ID')
    record = {'repository': 'https://github.com/ViCCo-Group/LAION-fMRI',
              'commit': revision, 'resolved_utc': datetime.now(timezone.utc).isoformat(),
              'files': []}
    # Pin before requesting any file; never mix moving-main versions.
    (destination/'source_revision.json').write_text(json.dumps(record, indent=2)+'\n')
    for name in FILES:
        url = f'https://raw.githubusercontent.com/ViCCo-Group/LAION-fMRI/{revision}/scripts/splits/{name}'
        payload = get_small(url, 4_000_000)
        path = destination/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        record['files'].append({'path': name, 'url': url, 'bytes': len(payload)})
        (destination/'source_revision.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def repeat_metadata(value, role):
    """Keep missing CSV plan separate from provider intent and actual trial rows."""
    match = re.fullmatch(r'([1-9][0-9]*)rep', value)
    if value == '' and role == 'development_unique':
        return {'n_reps_csv': '', 'planned_repeats': None, 'intended_repeats': 4,
                'intended_repeat_source': UNIQUE_REPEAT_DESIGN}
    if not match:
        raise ValueError('Unsupported planned-repeat metadata; only unique-image blanks are documented')
    number = int(match[1])
    if role == 'development_unique' and number != 4:
        raise ValueError('Unique-image CSV plan conflicts with provider four-repeat design')
    return {'n_reps_csv': value, 'planned_repeats': number, 'intended_repeats': number,
            'intended_repeat_source': 'source_stimulus_csv_n_reps'}


def read_images(path):
    images, roles = {}, Counter()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image_name', 'dataset', 'participant', 'unique_or_shared', 'n_reps'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing required stimulus metadata columns')
        for line, row in enumerate(reader, 2):
            image_id = row['image_name']
            if not image_id or image_id in images:
                raise ValueError(f'Duplicate or blank image identity at metadata line {line}')
            participants = row['participant'].split(';')
            if len(set(participants)) != len(participants) or not set(participants) <= set(PARTICIPANTS):
                raise ValueError(f'Unsupported participant field at metadata line {line}')
            if row['unique_or_shared'] == 'unique':
                if len(participants) != 1 or row['dataset'] not in SOURCE_NAMES:
                    raise ValueError(f'Unsupported unique-image role at metadata line {line}')
                role = 'development_unique'
            elif row['unique_or_shared'] == 'shared':
                if set(participants) != set(PARTICIPANTS):
                    raise ValueError(f'Shared metadata participant mismatch at line {line}')
                if row['dataset'] == 'OOD':
                    role = 'conditional_ood'
                elif row['dataset'] in {*SOURCE_NAMES, 'NSD'}:
                    role = 'excluded_regular_shared'
                else:
                    raise ValueError(f'Unknown shared source disposition at line {line}')
            else:
                raise ValueError(f'Unknown metadata role at line {line}')
            roles[role] += 1
            images[image_id] = {'image_id': image_id, 'participants': participants,
                                'role': role, 'stratum': SOURCE_NAMES.get(row['dataset']),
                                **repeat_metadata(row['n_reps'], role)}
    expected = {'development_unique': 23560, 'excluded_regular_shared': 1121,
                'conditional_ood': 371}
    if dict(roles) != expected:
        raise ValueError(f'Source image-role counts differ from current contract: {dict(roles)}')
    for participant in PARTICIPANTS:
        if sum(row['role'] == 'development_unique' and row['participants'] == [participant]
               for row in images.values()) != 4712:
            raise ValueError('Unique-image participant counts differ from current contract')
    return images, dict(roles)


def validate_groups(images, provider):
    """Preserve provider components and reject inconsistent duplicate corrections."""
    spec = json.loads((provider/'data/pooled_image_groups.json').read_text())
    if spec['cosine_distance_cutoffs'] != CUTOFFS:
        raise ValueError('Provider grouping thresholds differ from the adopted definition')
    regular = {key for key, row in images.items() if row['role'] != 'conditional_ood'}
    components, membership = [], {}
    for index, group in enumerate(spec['groups']):
        if not isinstance(group, list) or len(group) < 2 or len(set(group)) != len(group):
            raise ValueError('Invalid provider component member list')
        if not set(group) <= regular or set(group) & membership.keys():
            raise ValueError('Provider component has unknown/OOD/overlapping members')
        component_id = f'provider_component_{index:05d}'
        for image_id in group:
            membership[image_id] = component_id
        retained = sorted(key for key in group if images[key]['role'] == 'development_unique')
        if retained:
            components.append({'component_id': component_id, 'image_ids': retained,
                               'removed_shared_members': len(group)-len(retained)})
    pooled_duplicates = json.loads((provider/'data/pooled_duplicate_groups.json').read_text())
    by_pool = json.loads((provider/'data/duplicate_groups.json').read_text())
    changes = json.loads((provider/'data/random_fold_changes.json').read_text())
    if not isinstance(pooled_duplicates, list) or not set(by_pool) <= {*PARTICIPANTS, 'shared'}:
        raise ValueError('Unexpected provider duplicate metadata schema/pool')
    if not set(changes) <= {*PARTICIPANTS, 'shared'}:
        raise ValueError('Unknown corrected random-split pool')
    all_duplicate_groups = list(pooled_duplicates)
    all_duplicate_groups.extend(group for groups in by_pool.values() for group in groups)
    for group in all_duplicate_groups:
        if len(group) < 2 or len(set(group)) != len(group) or not set(group) <= regular:
            raise ValueError('Invalid duplicate group or nonregular member')
        if any(key not in membership for key in group) or len({membership[key] for key in group}) != 1:
            raise ValueError('Known duplicate group is not wholly in one provider component')
    # Recreate only provider random ID assignments to authenticate the correction
    # metadata. These are NEVER adopted as EP04 folds and read no feature values.
    for pool in ('shared', *PARTICIPANTS):
        ids = sorted(key for key in regular if images[key]['role'] == 'excluded_regular_shared'
                     or images[key]['participants'] == [pool])
        shuffled = np.array(ids, dtype=object)
        np.random.default_rng(42).shuffle(shuffled)
        splits = np.array_split(shuffled, 5)
        assignment = {str(key): fold for fold, part in enumerate(splits) for key in part}
        seen = set()
        for change in changes.get(pool, []):
            key, old, new = change['image_id'], change['old_fold'], change['new_fold']
            if key in seen or type(old) is not int or type(new) is not int or not 0 <= new < 5 or old == new:
                raise ValueError('Invalid or repeated provider fold correction')
            if assignment.get(key) != old:
                raise ValueError('Provider correction does not match seeded original assignment')
            seen.add(key)
            assignment[key] = new
        counts = Counter(assignment.values())
        if [counts[f] for f in range(5)] != [len(part) for part in splits]:
            raise ValueError('Provider corrections do not preserve original fold sizes')
        for group in by_pool.get(pool, []):
            if not set(group) <= set(ids) or len({assignment[key] for key in group}) != 1:
                raise ValueError('Corrected provider assignment splits a known duplicate group')
    for key in sorted(regular-membership.keys()):
        if images[key]['role'] == 'development_unique':
            components.append({'component_id': f'singleton:{key}', 'image_ids': [key],
                               'removed_shared_members': 0})
    return components, {'provider_components': len(spec['groups']),
                        'duplicate_sets_checked': len(all_duplicate_groups),
                        'random_corrections_checked': sum(map(len, changes.values())),
                        'corrected_random_folds_used_as_ep04_folds': False}


def assign_folds(images, components):
    """One deterministic group pass balancing all fifteen cell proportions."""
    totals = Counter((row['participants'][0], row['stratum']) for row in images.values()
                     if row['role'] == 'development_unique')
    if set(totals) != set(CELLS) or min(totals.values()) < 5:
        raise ValueError('All fifteen participant/source cells need structural fold support')
    rows = []
    covered = set()
    for component in components:
        members = component['image_ids']
        if not members or len(set(members)) != len(members) or covered & set(members):
            raise ValueError('Repeated or empty development component')
        if any(key not in images or images[key]['role'] != 'development_unique' for key in members):
            raise ValueError('Excluded or unknown image in development component')
        covered.update(members)
        counts = Counter((images[key]['participants'][0], images[key]['stratum']) for key in members)
        maximum = max(Fraction(value, totals[cell]) for cell, value in counts.items())
        rows.append((maximum, len(members), min(members), component, counts))
    if covered != {key for key, row in images.items() if row['role'] == 'development_unique'}:
        raise ValueError('Development components do not cover the exact unique-image set')
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    count_by_fold = [Counter() for _ in range(5)]
    fold_sizes = [0]*5
    assignments, group_folds = {}, {}
    for _, size, _, component, counts in rows:
        def criterion(fold):
            # Terms for other cells/folds are identical across the five choices.
            delta = math.fsum(((count_by_fold[fold][cell]+number)/totals[cell]-.2)**2
                             - (count_by_fold[fold][cell]/totals[cell]-.2)**2
                             for cell, number in sorted(counts.items()))
            return delta, fold_sizes[fold], fold
        chosen = min(range(5), key=criterion)
        count_by_fold[chosen].update(counts)
        fold_sizes[chosen] += size
        group_folds[component['component_id']] = chosen
        for key in component['image_ids']:
            assignments[key] = {'fold': chosen, 'component_id': component['component_id']}
    if any(count_by_fold[fold][cell] < 3 for fold in range(5) for cell in CELLS):
        raise ValueError('A frozen fold/source cell has fewer than three images; do not retune groups')
    balance = [{'participant': p, 'source_stratum': s, 'total': totals[(p, s)],
                'fold_counts': [counts[(p, s)] for counts in count_by_fold]}
               for p, s in CELLS]
    return assignments, group_folds, balance


def trial_rows(stream, session):
    """Preserve TSV row order as the session array axis and run-local indices."""
    reader = csv.DictReader(stream, delimiter='\t')
    if reader.fieldnames != ['session', 'run', 'beta_index', 'label']:
        raise ValueError('Trial-table header differs from declared provider interface')
    per_run, last_run, rows = Counter(), 0, 0
    for axis_index, row in enumerate(reader):
        if (None in row or any(value is None for value in row.values())
                or row['session'] != session or not row['beta_index'].isdigit()
                or not row['run'].isdigit()):
            raise ValueError(f'Invalid trial coordinates in {session} row {axis_index + 2}')
        run, beta = int(row['run']), int(row['beta_index'])
        if not 1 <= run <= 12 or run < last_run:
            raise ValueError('Trial runs must be ordered blocks within the provider range 01-12')
        if beta != per_run[run]:
            raise ValueError('Within-run beta indices must be unique, ordered and contiguous from zero')
        per_run[run] += 1
        last_run, rows = run, rows + 1
        yield row | {'beta_index': beta, 'session_beta_axis_index': axis_index}
    if not rows:
        raise ValueError('Empty source trial table')


def read_trials(source_root, images):
    receipt = json.loads((source_root/'provider_manifests/selected_objects.json').read_text())
    selected = Counter(row['Key'] for row in receipt)
    trials, role_counts, file_summaries = [], Counter(), []
    presentations = Counter()
    for participant in PARTICIPANTS:
        for number in range(1, 31):
            session = f'ses-{number:02d}'
            key = (f'derivatives/glmsingle-tedana/{participant}/{session}/func/'
                   f'{participant}_{session}_task-images_desc-SingletrialBetas_trials.tsv')
            if selected[key] != 1:
                raise ValueError('An exact selected trial table is absent/duplicated in acquisition receipt')
            run_counts, current_roles = Counter(), Counter()
            with (source_root/'dataset'/key).open(newline='') as stream:
                for row in trial_rows(stream, session):
                    run_counts[int(row['run'])] += 1
                    image = images.get(row['label'])
                    if image is None or participant not in image['participants']:
                        raise ValueError(f'Unknown/incompatible trial-image join in {participant}/{session} row {row["session_beta_axis_index"] + 2}')
                    role = image['role']
                    role_counts[(participant, role)] += 1
                    current_roles[role] += 1
                    if role == 'development_unique':
                        presentations[row['label']] += 1
                        trials.append({'participant': participant, 'session': session,
                                       'run': row['run'], 'beta_index': row['beta_index'],
                                       'session_beta_axis_index': row['session_beta_axis_index'],
                                       'image_id': row['label'],
                                       'source_stratum': image['stratum'],
                                       'repeat_index_in_source_order': presentations[row['label']],
                                       'neural_presentation_validity': 'not_assessed'})
            file_summaries.append({'participant': participant, 'session': session,
                                   'trial_file': key, 'rows': sum(run_counts.values()),
                                   'runs': [{'run': f'{run:02d}', 'rows': count,
                                             'beta_index_min': 0, 'beta_index_max': count - 1}
                                            for run, count in sorted(run_counts.items())],
                                   'run_local_indices_unique_ordered_contiguous': True,
                                   'session_axis_is_original_tsv_row_order': True,
                                   'role_counts': dict(current_roles)})
        print(json.dumps({'phase': 'trial_metadata', 'participant_completed': participant,
                          'files_completed': len(file_summaries)}), flush=True)
    return trials, presentations, file_summaries, [
        {'participant': p, 'role': role, 'trials': n}
        for (p, role), n in sorted(role_counts.items())]


def write_jsonl(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('outputs/config/development_metadata.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError('Preserve previous attempts; use a fresh output directory')
    started = time.monotonic()
    args.output_dir.mkdir(parents=True)
    report = {'analysis_id': 'source_readiness', 'job_id': os.environ['SLURM_JOB_ID'],
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'metadata_fold_freeze_complete': False, 'full_source_qualified': False,
              'neural_or_embedding_values_opened': False, 'raw_stimulus_values_opened': False,
              'audit_access_authorized': False, 'source_root': str(args.source_root)}
    try:
        config = json.loads(args.config.read_text())
        if (config['participants'] != list(PARTICIPANTS) or config['source_strata'] != list(STRATA)
                or config['fold_count'] != 5 or config['group_thresholds'] != CUTOFFS
                or config['unique_repeat_design']['intended_repeats'] != 4
                or config['unique_repeat_design']['source'] != UNIQUE_REPEAT_DESIGN):
            raise ValueError('Configuration differs from the adopted implementation')
        (args.output_dir/'adopted_config.json').write_text(json.dumps(config, indent=2)+'\n')
        report['provider'] = acquire_provider(args.output_dir/'provider_metadata', config.get('provider_commit'))
        print(json.dumps({'phase': 'provider_metadata_pinned',
                          'commit': report['provider']['commit']}), flush=True)
        images, report['image_role_counts'] = read_images(args.source_root/'dataset/stimuli/task-images_metadata.csv')
        components, report['group_checks'] = validate_groups(images, args.output_dir/'provider_metadata')
        trials, presentations, file_summaries, role_counts = read_trials(args.source_root, images)
        assignments, group_folds, balance = assign_folds(images, components)
        planned = Counter(row['intended_repeats'] for row in images.values() if row['role'] == 'development_unique')
        modes = [key for key, n in planned.items() if n == max(planned.values())]
        if len(modes) != 1:
            raise ValueError('Planned-repeat metadata has no unique mode; freeze unresolved')
        image_rows = []
        for key, image in sorted(images.items()):
            if image['role'] != 'development_unique':
                continue
            image_rows.append({'image_id': key, 'participant': image['participants'][0],
                               'source_stratum': image['stratum'], **assignments[key],
                               'planned_repeats': image['planned_repeats'],
                               'n_reps_csv': image['n_reps_csv'],
                               'intended_repeats': image['intended_repeats'],
                               'intended_repeat_source': image['intended_repeat_source'],
                               'observed_trial_metadata_rows': presentations[key],
                               'neural_valid_repeat_count': None})
        for trial in trials:
            trial.update(assignments[trial['image_id']])
        for component in components:
            component['fold'] = group_folds[component['component_id']]
        observed_repeats = Counter(presentations[row['image_id']] for row in image_rows)
        # Only publish derived views after every identity/group check has passed.
        write_jsonl(args.output_dir/'development_images.jsonl', image_rows)
        write_jsonl(args.output_dir/'development_trials.jsonl', trials)
        write_jsonl(args.output_dir/'development_groups.jsonl', components)
        report.update(metadata_fold_freeze_complete=True, trial_files_processed=len(file_summaries),
                      development_images=len(image_rows), development_trials=len(trials),
                      development_groups=len(components), fold_balance=balance,
                      planned_repeat_histogram=dict(planned), modal_planned_repeat_count=modes[0],
                      planned_repeat_basis=config['unique_repeat_design'],
                      blank_development_csv_repeat_fields=sum(row['planned_repeats'] is None for row in image_rows),
                      observed_development_repeat_histogram=dict(observed_repeats),
                      observed_trial_metadata_repeat_modes=sorted(value for value, count in observed_repeats.items()
                                                                  if count == max(observed_repeats.values())),
                      development_images_with_less_than_two_metadata_presentations=sum(presentations[row['image_id']] < 2 for row in image_rows),
                      trial_role_counts=role_counts, trial_files=file_summaries,
                      trial_axis_contract=config['trial_axis'],
                      role_disjointness='One metadata role per image; only unique-role trial/image rows emitted',
                      neural_eligibility='Unassessed; metadata rows are not finite/usable beta presentations',
                      source_model_checkpoint_binding='Pending later permitted feature metadata step',
                      remaining_required_inputs=['low_level_geometry', 'role_filtered_beta_and_feature_handoff',
                                                 'ROI_crosswalk', 'training_only_preprocessing', 'encoding_diagnostic'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Structural messages identify coordinates/counts, never source value dumps.
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['elapsed_seconds'] = time.monotonic()-started
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output_dir/'source_metadata_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('metadata_fold_freeze_complete', 'elapsed_seconds')}
                     | {'error': report.get('error'), 'output_dir': str(args.output_dir)}), flush=True)
    return 0 if report['metadata_fold_freeze_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
