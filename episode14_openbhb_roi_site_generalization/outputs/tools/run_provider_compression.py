#!/usr/bin/env python3
"""EP14 provider-group compression versus global PCA and fixed shuffled groups.

Exposed development only. All dimensions and shams are reported; no promotion.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
import run_development as base
from run_bilateral_regularization import PROVIDER_FAMILIES, check_deadline

EPISODE = Path(__file__).resolve().parents[2]
EXPOSED = EPISODE / 'inputs/openbhb/exposed'
ALPHAS = np.array([.1, 1., 10., 100., 1000., 10000., 100000.])
SEEDS = (14021, 14022, 14023)
LEVELS = {'anatomical': (1, 2, 4), 'measurement': (2, 4, 8)}
GROUP_COUNTS = {'anatomical': 34, 'measurement': 7}
VARIANTS = ('provider',) + tuple(f'shuffle{s}' for s in SEEDS)
SPEC = {
    'analysis_id': 'provider_group_compression', 'atlas': 'desikan',
    'participants': 3227, 'views': 3, 'alphas': ALPHAS.tolist(),
    'components_per_group': LEVELS, 'group_counts': GROUP_COUNTS,
    'shuffle_seeds': SEEDS, 'transform': 'fit_raw_scaling_then_unwhitened_PCA_no_postrotation_scaling',
    'tuning': 'seven_alphas_at_each_fixed_dimension_inner_leave_one_siteXacq_out_domain_macro_MAE',
    'tie_break': 'larger_alpha', 'primary': 'paired_external_domain_macro_MAE_provider_minus_matched_control',
    'holdout_access': False, 'candidate_promotion': False,
    'dimension_selection': False, 'original_disposition': 'development_no_candidate_unchanged',
    'uncertainty': '10000_paired_domain_then_participant_bootstraps_descriptive_uncorrected',
}


def arm_name(family, variant, k):
    return f'{family}_{variant}_k{k}'


def arm_menu():
    menu = {'raw': {'family': 'raw', 'dimension': 476}}
    for family, levels in LEVELS.items():
        for k in levels:
            dimension = GROUP_COUNTS[family] * k
            menu[f'pca{dimension}'] = {'family': 'global', 'dimension': dimension}
            for variant in VARIANTS:
                menu[arm_name(family, variant, k)] = {
                    'family': family, 'variant': variant, 'k': k, 'dimension': dimension}
    return menu


def provider_groups(names, labels, families):
    """Use only the exact provider label/channel cross-product, before values."""
    if tuple(families) != PROVIDER_FAMILIES or len(labels) != 68 or len(set(labels)) != 68:
        raise ValueError('Require all seven provider channels and 68 unique Desikan labels')
    regions = [label[3:] for label in labels if label.startswith('lh-')]
    if len(regions) != 34 or set(labels) != {f'{h}-{r}' for h in ('lh', 'rh') for r in regions}:
        raise ValueError('Provider labels do not define 34 exact bilateral ROIs')
    expected = {f'{label}_{family}' for label in labels for family in families}
    if len(names) != 476 or len(set(names)) != 476 or set(names) != expected:
        raise ValueError('Headers differ from the complete provider ROI/channel cross-product')
    lookup = {name: i for i, name in enumerate(names)}
    cube = np.array([[[lookup[f'{hemi}-{region}_{family}'] for family in families]
                      for hemi in ('lh', 'rh')] for region in regions])
    groups = {
        'anatomical': {'provider': cube.reshape(34, 14)},
        'measurement': {'provider': cube.transpose(2, 0, 1).reshape(7, 68)},
    }
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        shuffled = cube.copy()
        # Each anatomical sham keeps one feature of every hemisphere/channel.
        for hemisphere in range(2):
            for channel in range(7):
                shuffled[:, hemisphere, channel] = rng.permutation(cube[:, hemisphere, channel])
        groups['anatomical'][f'shuffle{seed}'] = shuffled.reshape(34, 14)
        rng = np.random.RandomState(seed)
        shuffled = cube.copy()
        # Each channel sham keeps one feature at every hemisphere/ROI location.
        # Mixing channels occurs only after common fit-only unit standardization.
        for region in range(34):
            for hemisphere in range(2):
                shuffled[region, hemisphere, :] = rng.permutation(cube[region, hemisphere, :])
        groups['measurement'][f'shuffle{seed}'] = shuffled.transpose(2, 0, 1).reshape(7, 68)
    return groups


def admitted_metadata():
    ref = EXPOSED / 'reference'
    labels = (ref / 'desikan_labels.txt').read_text().splitlines()
    families = (ref / 'freesurfer_channels.txt').read_text().splitlines()
    names = base.read_json(EXPOSED / 'feature_names_desikan.json')
    return names, provider_groups(names, labels, families)


def leading_basis(z, k):
    _, singular, vt = svd(z, full_matrices=False)
    tolerance = max(z.shape) * np.finfo(z.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < k:
        raise ValueError(f'Required PCA dimension {k} exceeds fit rank {rank}; no silent truncation')
    return vt[:k].T, singular


def representation_pairs(x_fit, x_held, groups, deadline):
    """Yield each fixed representation fitted using x_fit only."""
    scale = base.Scale().fit(x_fit)
    z, held = scale.transform(x_fit), scale.transform(x_held)
    center = z.mean(0)
    z, held = z - center, held - center
    yield 'raw', z, held, {'dimension': z.shape[1]}
    dimensions = sorted({GROUP_COUNTS[f] * k for f in LEVELS for k in LEVELS[f]})
    check_deadline(deadline)
    basis, singular = leading_basis(z, max(dimensions))
    fit_pc, held_pc = z @ basis, held @ basis
    energy = float(np.square(singular).sum())
    for dimension in dimensions:
        yield f'pca{dimension}', fit_pc[:, :dimension], held_pc[:, :dimension], {
            'dimension': dimension, 'training_variance_fraction': float(np.square(singular[:dimension]).sum()/energy)}
    for family, levels in LEVELS.items():
        maximum = max(levels)
        for variant in VARIANTS:
            check_deadline(deadline)
            fit_parts, held_parts, spectra = [], [], []
            for indices in groups[family][variant]:
                basis, singular = leading_basis(z[:, indices], maximum)
                fit_parts.append(z[:, indices] @ basis)
                held_parts.append(held[:, indices] @ basis)
                spectra.append(singular)
            for k in levels:
                yield arm_name(family, variant, k), np.column_stack([v[:, :k] for v in fit_parts]), np.column_stack([v[:, :k] for v in held_parts]), {
                    'dimension': GROUP_COUNTS[family]*k,
                    'training_variance_fraction': float(sum(np.square(s[:k]).sum() for s in spectra)/energy)}


def fit_view(view, out_string, deadline):
    with threadpool_limits(limits=1):
        start = time.monotonic()
        out = Path(out_string)
        names, groups = admitted_metadata()
        bundle = base.Bundle(EXPOSED)
        x, checked = bundle.atlas('desikan')
        if checked != names:
            raise ValueError('Provider feature headers changed')
        split, age, sites = bundle.splits[view], bundle.age, bundle.sites
        fit = split['fit']
        domains = np.unique(sites[fit])
        inner = []
        for number, domain in enumerate(domains):
            check_deadline(deadline)
            began = time.monotonic()
            train, valid = fit[sites[fit] != domain], fit[sites[fit] == domain]
            losses, diagnostics = {}, {}
            for arm, z, held, info in representation_pairs(x[train], x[valid], groups, deadline):
                check_deadline(deadline)
                predictions, rank = base.ridge_grid(z, age[train], held, ALPHAS)
                if arm != 'raw' and rank != info['dimension']:
                    raise ValueError(f'{arm}: deficient combined component rank {rank}')
                losses[arm] = np.mean(np.abs(predictions - age[valid, None]), 0).tolist()
                diagnostics[arm] = {**info, 'effective_rank': rank}
            if set(losses) != set(arm_menu()):
                raise ValueError('Incomplete inner comparison')
            record = {'view': view, 'domain': str(domain), 'train_rows': len(train),
                      'validation_rows': len(valid), 'losses': losses, 'diagnostics': diagnostics,
                      'seconds': time.monotonic() - began}
            base.write_json(out/'inner'/f'view{view}_domain{number}.json', record)
            inner.append(record)
            print(json.dumps({'stage': 'inner', 'view': view, 'domain': str(domain), 'seconds': record['seconds']}), flush=True)
        both = np.concatenate((split['internal'], split['external']))
        records = {}
        for arm, z, held, info in representation_pairs(x[fit], x[both], groups, deadline):
            check_deadline(deadline)
            loss = np.mean([item['losses'][arm] for item in inner], 0)
            selected = min(range(len(ALPHAS)), key=lambda i: (float(loss[i]), -float(ALPHAS[i])))
            alpha = float(ALPHAS[selected])
            prediction, rank = base.ridge_grid(z, age[fit], held, np.array([alpha]))
            if arm != 'raw' and rank != info['dimension']:
                raise ValueError(f'{arm}: deficient outer component rank')
            probe = base.fit_classifier(z, sites[fit])
            n_internal = len(split['internal'])
            site_pred = probe.predict(held[:n_internal])
            appearances = []
            for i, row in enumerate(both):
                item = {'id': str(bundle.ids[row]), 'domain': str(sites[row]), 'age': float(age[row]),
                        'prediction': float(prediction[i, 0]), 'absolute_error': float(abs(prediction[i, 0]-age[row])),
                        'role': 'internal' if i < n_internal else 'external', 'view': view}
                if i < n_internal:
                    item['site_correct'] = int(site_pred[i] == sites[row])
                appearances.append(item)
            records[arm] = {**info, 'view': view, 'arm': arm, 'alpha': alpha,
                            'effective_rank': rank, 'inner_domain_macro_mae': loss.tolist(),
                            'appearances': appearances}
            print(json.dumps({'stage': 'outer', 'view': view, 'arm': arm, 'alpha': alpha}), flush=True)
        if set(records) != set(arm_menu()):
            raise ValueError('Incomplete outer comparison')
        base.write_json(out/'outer'/f'view{view}.json', {'view': view, 'arms': records})
        return {'view': view, 'complete': True, 'seconds': time.monotonic()-start}


def joint_bootstrap(collapsed, deadline):
    """Same paired hierarchy for all arms, sharing draws across contrasts."""
    arms = list(arm_menu())
    reference = collapsed['raw']
    keys = sorted(reference)
    for arm in arms:
        if (sorted(collapsed[arm]) != keys or any(collapsed[arm][k]['domain'] != reference[k]['domain'] or collapsed[arm][k]['age'] != reference[k]['age'] for k in keys)):
            raise ValueError('Unpaired participant or domain support')
    groups = [np.array([[collapsed[arm][k]['loss'] for arm in arms] for k in keys if reference[k]['domain'] == domain])
              for domain in sorted({v['domain'] for v in reference.values()})]
    rng = np.random.RandomState(base.SEED)
    draws = np.empty((10000, len(arms)))
    for iteration in range(len(draws)):
        if iteration % 100 == 0:
            check_deadline(deadline)
        means = []
        for index in rng.randint(len(groups), size=len(groups)):
            group = groups[index]
            means.append(group[rng.randint(len(group), size=len(group))].mean(0))
        draws[iteration] = np.mean(means, 0)
    return {arm: draws[:, i] for i, arm in enumerate(arms)}


def aggregate(out, deadline):
    outer = [base.read_json(out/'outer'/f'view{view}.json') for view in range(3)]
    menu = arm_menu()
    records = {arm: [view['arms'][arm] for view in outer] for arm in menu}
    age = {arm: base.collapse(records[arm], 'external', 'absolute_error') for arm in menu}
    results = {arm: {
        **menu[arm], 'external_age': base.summarize(age[arm]),
        'internal_age': base.summarize(base.collapse(records[arm], 'internal', 'absolute_error')),
        'internal_site': base.summarize(base.collapse(records[arm], 'internal', 'site_correct')),
        'per_view_external': [base.summarize(base.collapse([r], 'external', 'absolute_error')) for r in records[arm]],
        'selected_alpha': [r['alpha'] for r in records[arm]],
        'effective_rank': [r['effective_rank'] for r in records[arm]],
        'training_variance_fraction': [r.get('training_variance_fraction', 1.) for r in records[arm]],
    } for arm in menu}
    draws = joint_bootstrap(age, deadline)
    contrasts = []
    for family, levels in LEVELS.items():
        for k in levels:
            arm = arm_name(family, 'provider', k)
            dimension = menu[arm]['dimension']
            for reference in ('raw', f'pca{dimension}', *(arm_name(family, v, k) for v in VARIANTS[1:])):
                ref, alt = results[reference]['external_age'], results[arm]['external_age']
                difference = draws[arm] - draws[reference]
                delta = alt['domain_macro'] - ref['domain_macro']
                contrasts.append({'family': family, 'k': k, 'dimension': dimension, 'arm': arm, 'reference': reference,
                    'delta_age': delta, 'relative_age_change': delta/ref['domain_macro'],
                    'descriptive_tailed_95_interval': np.quantile(difference, [.025, .975]).tolist(),
                    'domain_changes': {d: v['mean'] - ref['domains'][d]['mean'] for d, v in alt['domains'].items()},
                    'leave_one_domain_out_changes': {d: v-ref['leave_one_domain_out'][d] for d, v in alt['leave_one_domain_out'].items()},
                    'delta_participant_pooled': alt['participant_pooled'] - ref['participant_pooled'],
                    'delta_worst_domain': alt['worst_domain'] - ref['worst_domain'],
                    'delta_site_BAcc': results[arm]['internal_site']['domain_macro']-results[reference]['internal_site']['domain_macro']})
    report = {**SPEC, 'complete': True, 'results': results, 'contrasts': contrasts,
              'arm_count': len(menu), 'bootstrap_seed': base.SEED,
              'claim': 'Exploratory exposed-development compression comparison; no dimension selection, candidate promotion, multiplicity-adjusted inference, biological mechanism or independent replication.'}
    base.write_json(out/'provider_compression_report.json', report)
    lines = ['# Provider-group compression: exposed development only', '',
             'All three views, six fixed compression dimensions, 24 grouped arms, six global-PCA controls and raw ridge are retained.',
             'All 757 public-validation rows remain closed; the original no-candidate result is unchanged.', '',
             '| Arm | Dimensions | Domain MAE | Participant MAE | Worst-domain MAE | Oldest-quartile MAE | Site BAcc |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in results.items():
        a = row['external_age']
        oldest = a['age_quartiles'][-1]['mean']
        lines.append(f"| {arm} | {row['dimension']} | {a['domain_macro']:.4f} | {a['participant_pooled']:.4f} | {a['worst_domain']:.4f} | {oldest:.4f} | {row['internal_site']['domain_macro']:.4f} |")
    lines += ['', 'Paired provider-minus-reference differences (years; negative favors provider groups):', '',
              '| Family | Dimension | Reference | Difference | Descriptive 95% interval |', '|---|---:|---|---:|---|']
    for row in contrasts:
        low, high = row['descriptive_tailed_95_interval']
        lines.append(f"| {row['family']} | {row['dimension']} | {row['reference']} | {row['delta_age']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ['', 'Intervals use 10,000 paired domain/participant resamples and are not multiplicity adjusted. Overlapping views are not independent cohorts.',
              'Dimensions are output dimensions; covariance fitting complexity differs between global and block PCA and is not equated by dimension matching.',
              'All fitting, including scaling, PCA and alpha tuning, uses the corresponding training role only. No whitening or post-PCA rescaling.',
              'Each sham is retained individually. Anatomical claims require comparison against both same-size block shams and global PCA; a raw-ridge win alone is insufficient.', '']
    (out/'provider_compression_report.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-seconds', type=int, default=3300)
    args = parser.parse_args()
    base.compute_only()
    if not 1 <= args.workers <= min(3, int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))):
        raise ValueError('Require 1–3 workers within allocation')
    if not 1 <= args.max_seconds <= 3300:
        raise ValueError('Use a runtime bound of at most 3300 seconds')
    out = args.out.resolve()
    if not out.is_relative_to(EPISODE/'outputs/runs/provider_group_compression') or out.exists():
        raise ValueError('Use a fresh episode-local provider_group_compression attempt directory')
    started = time.monotonic()
    deadline = started + args.max_seconds
    names, groups = admitted_metadata()
    out.mkdir(parents=True)
    base.write_json(out/'specification.json', SPEC)
    base.write_json(out/'provider_groups.json', {'feature_names': names,
        'groups': {f: {v: g.tolist() for v, g in values.items()} for f, values in groups.items()}})
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit_view, view, str(out), deadline): view for view in range(3)}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {'view': futures[future], 'complete': False, 'error': f'{type(exc).__name__}: {exc}'}
            completed.append(result)
            print(json.dumps(result), flush=True)
    status = {'analysis_id': SPEC['analysis_id'], 'views': completed, 'complete': False, 'holdout_access': False}
    if len(completed) == 3 and all(row['complete'] for row in completed):
        try:
            with threadpool_limits(limits=1):
                aggregate(out, deadline)
            status['complete'] = True
        except Exception as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
    status['seconds'] = time.monotonic()-started
    base.write_json(out/'run_status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
