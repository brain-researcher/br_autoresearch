#!/usr/bin/env python3
"""Collect the complete exploratory EP01 cohort without complete-case filtering.

Consumes prepared metadata, completed beta products and an outcome-blind mask;
never reads BOLD. Execute numerical work only in a verified Slurm allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import time

import numpy as np

from exploratory_fmri_runner import ARMS, FOCAL, atomic_json, require_compute

EPISODE = Path(__file__).resolve().parents[2]
MEMBERSHIP = EPISODE / 'outputs/runs/public_asset_qualification/20261003-schedule01/offer_sequence_membership.tsv'
AXES = ('S11', 'total_displacement', 'decision_omission', 'response_omission',
        'interaction', 'response_deadline_timing')
WEIGHTS = np.array([[0, 0, 0, 1, 0], [1, 0, 0, -1, 0],
                    [.5, -.5, .5, -.5, 0], [.5, .5, -.5, -.5, 0],
                    [1, -1, -1, 1, 0], [0, 0, 0, -1, 1]], dtype=float)
PANELS = ('half14', 'half23', 'all4')
STRATA = ('all', 'EI', 'ER')


def axis_maps(effects):
    """Do not let a missing, zero-weight arm contaminate an unrelated axis."""
    return np.stack([np.tensordot(weight[weight != 0], effects[weight != 0], axes=(0, 0))
                     for weight in WEIGHTS])


def scalar_summary(values):
    """A required missing participant makes the requested panel unavailable."""
    from scipy.stats import t as student_t
    values = np.asarray(values, dtype=float)
    available = np.isfinite(values)
    result = {'expected_participants': len(values), 'available_participants': int(available.sum()),
              'unavailable_participants': int((~available).sum()), 'mean': None,
              'participant_SE': None, 't95_CI': None,
              'status': 'unavailable_without_complete_case_filtering'}
    if len(values) < 2 or not available.all():
        return result
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(student_t.ppf(.975, len(values)-1) * se)
    result.update(status='complete', mean=mean, participant_SE=se,
                  t95_CI=[mean-margin, mean+margin])
    return result


def cosine(left, right):
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float('nan')
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(np.dot(left, right)/denominator, -1., 1.)) if denominator > 0 else float('nan')


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if np.isfinite(values).all() else float('nan')


def ratio(numerator, denominator):
    return numerator/denominator if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0 else float('nan')


def participant_metrics(halves):
    """Input is (2 halves, 5 arms, 2 focal slopes, common voxels)."""
    panels = [halves[0], halves[1], .5*(halves[0]+halves[1])]
    axes = [axis_maps(panel) for panel in panels]
    result = []
    for ai, axis in enumerate(AXES):
        for fi, focal in enumerate(FOCAL):
            result.append({'axis': axis, 'focal': focal, 'panel': 'half14_vs_half23',
                           'metric': 'uncentered_spatial_cosine',
                           'value': cosine(axes[0][ai, fi], axes[1][ai, fi])})
            for pi, panel in enumerate(PANELS):
                result.append({'axis': axis, 'focal': focal, 'panel': panel,
                               'metric': 'map_RMS', 'value': rms(axes[pi][ai, fi])})
    for pi, panel in enumerate(PANELS):
        for fi, focal in enumerate(FOCAL):
            baseline, full = rms(panels[pi][0, fi]), rms(panels[pi][3, fi])
            displacement = rms(axes[pi][1, fi])
            for metric, value in (
                ('baseline_S00_RMS', baseline), ('full_S11_RMS', full),
                ('displacement_RMS_over_baseline_RMS', ratio(displacement, baseline)),
                ('displacement_RMS_over_full_RMS', ratio(displacement, full)),
                ('displacement_RMS_over_symmetric_mean_RMS', ratio(displacement, .5*(baseline+full))),
                ('full_RMS_over_baseline_RMS', ratio(full, baseline))):
                result.append({'axis': 'total_displacement', 'focal': focal, 'panel': panel,
                               'metric': metric, 'value': value})
    return axes, result


def validate_inventory(inventory):
    participants = inventory.get('participants', [])
    runs = inventory.get('runs', [])
    if inventory.get('status') != 'complete' or len(participants) != 108 or len(set(participants)) != 108 or len(runs) != 432:
        raise ValueError('Require completed preparation of all 108 participants and 432 runs')
    entries, versions = {}, {}
    for entry in runs:
        person, run = entry['participant'], entry['run']
        if person not in participants or run not in (1, 2, 3, 4) or (person, run) in entries:
            raise ValueError('Prepared participant/run support is not one-to-one')
        if entry['task_version'] not in ('EI', 'ER') or versions.setdefault(person, entry['task_version']) != entry['task_version']:
            raise ValueError('Invalid or changing task version')
        entries[person, run] = entry
    if set(entries) != {(p, r) for p in participants for r in (1, 2, 3, 4)} or Counter(versions.values()) != {'EI': 54, 'ER': 54}:
        raise ValueError('Require all four runs and 54 participants per task version')
    return sorted(participants), entries, versions


def collect_completions(prepared, roots, entries, check=lambda: None):
    completed, missing = {}, []
    for key, entry in entries.items():
        check()
        name = f'{key[0]}_run-{key[1]}'
        paths = [root/name/'completion.json' for root in roots if (root/name/'completion.json').is_file()]
        if len(paths) > 1:
            raise ValueError(f'Ambiguous duplicate completed run: {name}')
        if not paths:
            missing.append({'participant': key[0], 'run': key[1]})
            continue
        completion = json.loads(paths[0].read_text())
        if (completion.get('status') != 'complete' or completion.get('participant') != key[0]
                or completion.get('run') != key[1] or completion.get('task_version') != entry['task_version']
                or completion.get('prepared') != str(prepared.resolve())
                or completion.get('designs') != entry['designs']
                or completion.get('design_diagnostics') != entry['design_diagnostics']):
            raise ValueError(f'Completion does not match the prepared run: {name}')
        artifact = paths[0].parent/'gain_loss_effects.npz'
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        completed[key] = (artifact, completion)
    return completed, missing


def load_memberships(path, versions):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['participant', 'task_version', 'offer_sequence_id']:
            raise ValueError('Use the qualified design-only offer-sequence membership table')
        rows = list(reader)
    memberships = {}
    expected_ids = {f'{version}-offers-{number:02d}' for version in ('EI', 'ER') for number in range(1, 9)}
    for row in rows:
        person, group = row['participant'], row['offer_sequence_id']
        if person in memberships or person not in versions or row['task_version'] != versions[person] or group not in expected_ids or not group.startswith(versions[person]+'-'):
            raise ValueError('Offer-sequence membership does not match the full prepared cohort')
        memberships[person] = group
    if set(memberships) != set(versions) or set(memberships.values()) != expected_ids:
        raise ValueError('Require all 108 memberships and all 16 offer-sequence groups')
    return memberships


def common_positions(indices, common_indices):
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or (np.diff(indices) <= 0).any():
        raise ValueError('Saved voxel indices must be unique increasing integers')
    positions = np.searchsorted(indices, common_indices)
    if (positions >= len(indices)).any() or not np.array_equal(indices[positions], common_indices):
        raise ValueError('A run lacks voxels in the fixed group mask; no support filtering is allowed')
    return positions


def load_effects(path, completion, common_indices, shape, affine):
    with np.load(path, allow_pickle=False) as source:
        if tuple(source['arms'].tolist()) != ARMS or tuple(source['focal'].tolist()) != FOCAL:
            raise ValueError('Saved arm/focal order differs from the fixed analysis')
        if tuple(source['shape']) != shape or not np.allclose(source['affine'], affine, rtol=0, atol=1e-5):
            raise ValueError('Saved effect and fixed group-mask grids differ')
        indices = source['voxel_indices']
        positions = common_positions(indices, common_indices)
        effects = source['effects']
        if effects.shape != (5, 2, len(indices)) or (indices < 0).any() or (indices >= np.prod(shape)).any():
            raise ValueError('Unexpected effect dimensions or voxel indices')
        result = effects[:, :, positions].astype(float)
    for ai, arm in enumerate(ARMS):
        estimable = completion['fit_diagnostics'][arm]['focal_estimable']
        for fi, focal in enumerate(FOCAL):
            values = result[ai, fi]
            if estimable[focal] is True and not np.isfinite(values).all():
                raise ValueError(f'Unexpected nonfinite estimable {arm}/{focal} coefficients')
            if estimable[focal] is False and not np.isnan(values).all():
                raise ValueError('Nonestimable coefficients must be explicitly unavailable')
            if not isinstance(estimable[focal], bool):
                raise ValueError('Missing focal estimability decision')
    return result


class MapMoments:
    """Welford moments; incomplete panels are never exported as group estimates."""
    def __init__(self, voxels):
        self.n = 0
        self.mean = np.zeros(voxels, dtype=float)
        self.m2 = np.zeros(voxels, dtype=float)

    def add(self, values):
        if not np.isfinite(values).all():
            if not np.isnan(values).all():
                raise ValueError('A map is partially unavailable; do not change voxel support')
            return
        self.n += 1
        delta = values-self.mean
        self.mean += delta/self.n
        self.m2 += delta*(values-self.mean)

    def estimates(self, expected):
        if self.n != expected or expected < 2:
            return None
        return self.mean, np.sqrt(np.maximum(self.m2, 0)/(expected-1)/expected)


def group_metric_summaries(records, versions, memberships):
    grouped = {}
    for row in records:
        key = tuple(row[name] for name in ('axis', 'focal', 'panel', 'metric'))
        grouped.setdefault(key, {})[row['participant']] = row['value']
    summaries, omissions = [], []
    for key, values in sorted(grouped.items()):
        if set(values) != set(versions):
            raise ValueError('A participant metric row is missing from the complete report')
        identity = dict(zip(('axis', 'focal', 'panel', 'metric'), key))
        for stratum in STRATA:
            people = [p for p in versions if stratum == 'all' or versions[p] == stratum]
            summaries.append({**identity, 'stratum': stratum,
                              **scalar_summary([values[p] for p in people])})
        for group in sorted(set(memberships.values())):
            for stratum in ('all', group[:2]):
                people = [p for p in versions if (stratum == 'all' or versions[p] == stratum) and memberships[p] != group]
                omissions.append({**identity, 'stratum': stratum, 'omitted_offer_sequence': group,
                                  'omitted_participants': sum(memberships[p] == group for p in versions),
                                  **scalar_summary([values[p] for p in people])})
    return summaries, omissions


def safe_records(records):
    return [{key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
             for key, value in row.items()} for row in records]


def save_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(safe_records(records))


def quality_report(entries, versions, check):
    failures = []
    for (person, run), entry in entries.items():
        check()
        diagnostics = json.loads(Path(entry['design_diagnostics']).read_text())['arms']
        for arm in ARMS:
            record = diagnostics[arm]
            checks = {'task_condition_pass': record['task_condition_pass'],
                      'focal_vif_pass': record['focal_vif_pass'],
                      'gain_estimable': record['focal_estimable']['gain'],
                      'loss_estimable': record['focal_estimable']['loss']}
            if arm in ('S11', 'R11'):
                checks['relative_efficiency_pass'] = record['relative_efficiency_pass']
            failed = [name for name, passed in checks.items() if passed is not True]
            if failed:
                failures.append({'participant': person, 'task_version': versions[person],
                                 'run': run, 'arm': arm, 'failed': failed})
    return {'run_arm_failures': failures, 'failed_run_arm_count': len(failures),
            'participants_with_any_failed_gate': sorted({row['participant'] for row in failures}),
            'counts_by_arm': dict(Counter(row['arm'] for row in failures)),
            'interpretation': 'Quality failures remain visible. Finite coefficients are exploratory descriptions; failed original gates do not support their original specificity/identification claim.'}


def write_plot(output, summaries):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {'all': '#555555', 'EI': '#2768a3', 'ER': '#bd6727'}
    for axis, metric, panel, chart in (
        ('total_displacement', 'uncentered_spatial_cosine', 'half14_vs_half23', panels[0]),
        ('total_displacement', 'displacement_RMS_over_symmetric_mean_RMS', 'all4', panels[1])):
        for si, stratum in enumerate(STRATA):
            for fi, focal in enumerate(FOCAL):
                row = next(r for r in summaries if r['axis'] == axis and r['metric'] == metric and r['panel'] == panel and r['stratum'] == stratum and r['focal'] == focal)
                position = fi + (si-1)*.18
                if row['status'] == 'complete':
                    lower, upper = row['t95_CI']
                    chart.errorbar(position, row['mean'], yerr=[[row['mean']-lower], [upper-row['mean']]],
                                   fmt='o', color=colors[stratum], capsize=3,
                                   label=stratum if fi == 0 else None)
                else:
                    chart.text(position, .03, 'NA', transform=chart.get_xaxis_transform(), ha='center', fontsize=7, color=colors[stratum])
        chart.set_xticks([0, 1], ['Gain', 'Positive loss'])
        chart.spines[['top', 'right']].set_visible(False)
        chart.axhline(0, color='#cccccc', linewidth=.8)
    panels[0].set_title('Cross-half agreement of S00 − S11')
    panels[0].set_ylabel('Mean participant spatial cosine; t 95% CI')
    panels[1].set_title('Magnitude of S00 − S11')
    panels[1].set_ylabel('RMS displacement / mean baseline and full RMS')
    if panels[0].get_legend_handles_labels()[0]:
        panels[0].legend(frameon=False)
    fig.suptitle('Exploratory NARPS reanalysis · participants are the sampling units', fontsize=11)
    fig.tight_layout()
    paths = []
    for extension in ('png', 'pdf'):
        path = output/f'exploratory_fmri_summary.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(path.name)
    plt.close(fig)
    return {'status': 'complete', 'files': paths, 'intervals': 'uncorrected participant t95 intervals; descriptive, not voxel tests or original equivalence decisions'}


def summarize(prepared, roots, output, membership, check, approved_support=None):
    import nibabel as nib
    from exploratory_fmri_support import selected_support
    if output.exists():
        raise FileExistsError('Use a fresh summary output directory')
    inventory = json.loads((prepared/'prepared.json').read_text())
    participants, entries, versions = validate_inventory(inventory)
    completions, missing = collect_completions(prepared, roots, entries, check)
    output.mkdir(parents=True)
    report = {'analysis_id': 'exploratory_fmri_summary', 'status': 'incomplete_cohort',
              'prepared': str(prepared.resolve()), 'run_roots': [str(root.resolve()) for root in roots],
              'expected_participants': 108, 'expected_runs': 432, 'complete_runs': len(completions),
              'missing_runs': missing, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'spatial_support':({'selection':'original_frozen_support'} if approved_support is None else
                  {'selection':'approved_valid_common_support_requested',
                   'approved_support':str(Path(approved_support).resolve()),
                   'status':'not_applied_incomplete_cohort'})}
    if missing:
        atomic_json(output/'exploratory_fmri_summary.json', report)
        (output/'exploratory_fmri_summary.md').write_text(f"Only {len(completions)} of 432 runs are complete. No full-cohort scientific summary, maps or metrics were produced.\n")
        return report
    selected_mask,selected_voxels,support_metadata=selected_support(prepared,inventory,approved_support)
    report['spatial_support']=support_metadata
    memberships = load_memberships(membership, versions)
    permitted_masks={Path(inventory['group_mask']).resolve(),selected_mask}
    for _,completion in completions.values():
        if Path(completion['group_mask']).resolve() not in permitted_masks:
            raise ValueError('Completed run used neither original nor explicitly selected support')
    mask_image = nib.load(selected_mask)
    mask_values = np.asanyarray(mask_image.dataobj)
    if mask_values.ndim != 3 or not np.isin(mask_values, [0, 1]).all():
        raise ValueError('Invalid frozen group mask')
    mask = mask_values == 1
    common_indices = np.flatnonzero(mask)
    if not len(common_indices) or len(common_indices) != selected_voxels:
        raise ValueError('Frozen group-mask voxel count changed')
    report['design_quality'] = quality_report(entries, versions, check)
    moments, records = {}, []
    for person in participants:
        check()
        halves = np.zeros((2, 5, 2, len(common_indices)), dtype=float)
        for run in (1, 2, 3, 4):
            check()
            artifact, completion = completions[person, run]
            effects = load_effects(artifact, completion, common_indices, mask.shape, mask_image.affine)
            halves[0 if run in (1, 4) else 1] += .5*effects
        axes, metrics = participant_metrics(halves)
        for row in metrics:
            records.append({'participant': person, 'task_version': versions[person],
                            'offer_sequence_id': memberships[person], **row})
        for pi, panel in enumerate(PANELS):
            for ai, axis in enumerate(AXES):
                for fi, focal in enumerate(FOCAL):
                    for stratum in ('all', versions[person]):
                        key = stratum, panel, axis, focal
                        if key not in moments:
                            moments[key] = MapMoments(len(common_indices))
                        moments[key].add(axes[pi][ai, fi])
        print(json.dumps({'stage': 'summarized_participant', 'participant': person}), flush=True)
    summaries, omissions = group_metric_summaries(records, versions, memberships)
    maps = []
    map_dir = output/'maps'
    map_dir.mkdir()
    for (stratum, panel, axis, focal), moment in sorted(moments.items()):
        check()
        expected = 108 if stratum == 'all' else 54
        result = moment.estimates(expected)
        record = {'stratum': stratum, 'panel': panel, 'axis': axis, 'focal': focal,
                  'expected_participants': expected, 'available_participants': moment.n,
                  'unavailable_participants': expected-moment.n,
                  'status': 'unavailable_without_complete_case_filtering', 'files': {}}
        if result is not None:
            record['status'] = 'complete'
            for statistic, values in zip(('participant_mean', 'participant_SE'), result):
                volume = np.full(mask.shape, np.nan, dtype='float32')
                volume.ravel()[common_indices] = values
                filename = f'{stratum}_{panel}_{axis}_{focal}_{statistic}.nii.gz'
                header = mask_image.header.copy()
                header.set_data_dtype('float32')
                header['descrip'] = b'EP01 descriptive participant mean/SE; outside fixed mask is NaN'
                nib.save(nib.Nifti1Image(volume, mask_image.affine, header), map_dir/filename)
                record['files'][statistic] = 'maps/'+filename
        maps.append(record)
    save_csv(output/'participant_metrics.csv', records)
    save_csv(output/'offer_sequence_omissions.csv', omissions)
    report.update(status='complete_cohort', participants=108, task_versions={'EI': 54, 'ER': 54},
                  fixed_group_mask=str(selected_mask), group_voxels=len(common_indices),
                  offer_sequence_membership=str(membership.resolve()),
                  axes={axis: dict(zip(ARMS, weight.tolist())) for axis, weight in zip(AXES, WEIGHTS)},
                  metric_summaries=summaries, group_maps=maps,
                  participant_metrics='participant_metrics.csv', offer_sequence_omissions='offer_sequence_omissions.csv',
                  interpretation={
                      'scope': 'Exploratory within-session conditional-estimand sensitivity; no DDM mechanism, neural-process identification, independent replication, team-map or corrected voxel-inference claim.',
                      'sampling_unit': 'participant; equal participant weights; half14 and half23 are paired repeated observations',
                      'task_versions': 'EI and ER are descriptive between-participant strata with different offer ranges; their contrast is not a causal task-version effect.',
                      'spatial_cosine': 'uncentered dot product divided by map L2 norms on the explicitly selected fixed common support; zero-norm maps are unavailable',
                      'intervals': 'Student t95 confidence intervals for means of paired-within-participant derived scalars; no voxel independence assumption and no multiplicity correction',
                      'ratios': 'Participant-wise RMS denominators explicitly named. Nonlinear RMS is computed after within-participant run averaging. These descriptive ratios are not original equivalence tests.',
                      'missingness': 'Any required nonestimable arm propagates to dependent participant maps/metrics; the corresponding all-participant stratum result is unavailable, never a complete-case estimate.',
                      'offer_omissions': 'Fixed-prediction equal-participant means after omitting each of 16 qualified version-specific offer groups, for all and its matching version. Overlapping descriptive influence subsets; no refits or template-generalization claim.',
                      'maps': 'Unthresholded means and participant standard errors; outside the fixed group mask is NaN. No inferential t/z/p maps.'},
                  completed_utc=datetime.now(timezone.utc).isoformat())
    report['figure'] = write_plot(output, summaries)
    atomic_json(output/'exploratory_fmri_summary.json', report)
    lines = ['# Exploratory NARPS conditional-estimand sensitivity', '',
             'All **108 participants × four runs** are present (54 EI, 54 ER). The summaries retain participant pairing and fixed common spatial support.', '',
             f"Design-quality failures affect **{len(report['design_quality']['participants_with_any_failed_gate'])} participants** and **{report['design_quality']['failed_run_arm_count']} run/arm combinations**. Finite estimates remain exploratory; failed gates do not establish the original specificity claim.", '',
             '| Stratum | Slope | Cross-half displacement cosine, mean [t95 CI] | Symmetric relative displacement, mean [t95 CI] |',
             '|---|---|---|---|']
    if approved_support is not None:
        lines[4:4]=[f"The spatial support was amended with scientist approval after observed input-validity failures: **{selected_voxels:,}** of **{support_metadata['original_voxels']:,}** original common-mask voxels are finite at every frame and have a positive run mean in all 432 runs. The original-support complete-cohort analysis remains unavailable. All summaries here use this one amended support.",'']
    def formatted(row):
        if row['status'] != 'complete':
            return f"Unavailable ({row['available_participants']}/{row['expected_participants']} available)"
        low, high = row['t95_CI']
        return f"{row['mean']:.4f} [{low:.4f}, {high:.4f}]"
    for stratum in STRATA:
        for focal in FOCAL:
            rows = [r for r in summaries if r['stratum'] == stratum and r['focal'] == focal and r['axis'] == 'total_displacement']
            cross = next(r for r in rows if r['metric'] == 'uncentered_spatial_cosine')
            relative = next(r for r in rows if r['metric'] == 'displacement_RMS_over_symmetric_mean_RMS' and r['panel'] == 'all4')
            lines.append(f"| {stratum} | {focal} | {formatted(cross)} | {formatted(relative)} |")
    lines += ['', 'Intervals describe participant means without multiplicity correction. Spatial voxels are not treated as independent observations. These ratios do not test the original equivalence margins.', '',
              '[Full report and all six contrasts](exploratory_fmri_summary.json) · [Participant metrics](participant_metrics.csv) · [Offer-sequence influence](offer_sequence_omissions.csv)', '',
              'A required nonestimable arm makes its dependent full-stratum summaries unavailable. No participant or voxel subset is selected to recover a positive result. Team maps remain closed; this analysis does not identify a psychological or temporal neural mechanism.', '']
    if report['figure']['status'] == 'complete':
        lines += ['![Exploratory participant summaries](exploratory_fmri_summary.png)', '']
    (output/'exploratory_fmri_summary.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--offer-membership', type=Path, default=MEMBERSHIP)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--max-rss-gib', type=float, default=12.)
    parser.add_argument('--approved-support',type=Path)
    args = parser.parse_args()
    require_compute()
    if args.max_seconds <= 0 or args.max_rss_gib <= 0:
        raise ValueError('Positive bounded runtime and RSS required')
    roots = [root.resolve() for root in args.run_root]
    if len(set(roots)) != len(roots):
        raise ValueError('Run roots must be unique')
    began = time.monotonic()
    def check():
        if time.monotonic()-began >= args.max_seconds:
            raise TimeoutError('Summary runtime limit reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > args.max_rss_gib:
            raise MemoryError('Summary RSS limit reached')
    def interrupted(signum, frame):
        raise TimeoutError(f'Summary interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(args.max_seconds)
    try:
        result = summarize(args.prepared_dir, roots, args.output_dir, args.offer_membership, check,args.approved_support)
        check()
        print(json.dumps({'status': result['status'], 'complete_runs': result['complete_runs'],
                          'elapsed_seconds': time.monotonic()-began}), flush=True)
        return 0 if result['status'] == 'complete_cohort' else 2
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
