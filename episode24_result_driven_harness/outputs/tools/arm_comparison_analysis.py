#!/usr/bin/env python3
"""EP24 arm comparison: the prespecified analysis.

Implements the `analysis` and `decision` sections of SEARCH_POLICY.yaml for one
attempt, with the 2026-10-16 rating amendment: the scientist's blind check was
waived and a second model rater added before unsealing. Each proposal's score
averages the model raters. Each rater alone is a sensitivity analysis, and
Cohen's kappa between the two raters replaces the human agreement check.
Writes analysis.json, per_item.csv and the figure.

Modes:
  run       analyze an attempt
  selftest  check the statistics on synthetic numbers; reads no run files
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

# Thresholds from SEARCH_POLICY.yaml (decision, rating.primary_scores).
MARGIN = 0.25
ALPHA = 0.05
KAPPA_MIN = 0.6
R4_TOLERANCE = 0.5
ARMS = ('old', 'new', 'directive')
METRICS = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')
SCALE = 720  # covers draw/rater means (1/12) averaged over up to 4 items per episode


def signflip_p(diffs):
    """Exact two-sided sign-flip p-value for the mean of item differences."""
    ints = [round(d * SCALE) for d in diffs]
    if any(abs(i / SCALE - d) > 1e-9 for i, d in zip(ints, diffs)):
        raise ValueError('item differences are not multiples of 1/12')
    dist = {0: 1}
    for v in ints:
        step = {}
        for s, c in dist.items():
            step[s + v] = step.get(s + v, 0) + c
            step[s - v] = step.get(s - v, 0) + c
        dist = step
    observed = abs(sum(ints))
    return sum(c for s, c in dist.items() if abs(s) >= observed) / 2 ** len(ints)


def inversion_interval(diffs, step=0.001):
    """95% interval for the mean shift: every delta the sign-flip test keeps."""
    n = len(diffs)
    patterns = list(itertools.product((1, -1), repeat=n))
    a = [sum(s * d for s, d in zip(p, diffs)) for p in patterns]
    b = [sum(p) for p in patterns]
    total = sum(diffs)
    kept = []
    for k in range(-round(1 / step), round(1 / step) + 1):
        delta = k * step
        observed = abs(total - n * delta) - 1e-9
        hits = sum(abs(ai - delta * bi) >= observed for ai, bi in zip(a, b))
        if hits / len(patterns) >= ALPHA:
            kept.append(k)
    if not kept:
        return None
    contiguous = kept == list(range(kept[0], kept[-1] + 1))
    return {'low': round(kept[0] * step, 3), 'high': round(kept[-1] * step, 3), 'contiguous': contiguous}


def kappa(x, y):
    """Cohen's kappa and raw agreement for two equal-length label lists."""
    n = len(x)
    if n == 0:
        return {'n': 0, 'kappa': None, 'agreement': None}
    agree = sum(a == b for a, b in zip(x, y)) / n
    expected = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    k = None if expected == 1 else (agree - expected) / (1 - expected)
    return {'n': n, 'kappa': None if k is None else round(k, 4), 'agreement': round(agree, 4)}


def load_ratings(run, dirname):
    out = {}
    for f in (run / dirname).glob('p*.json'):
        if '.failed' not in f.name:
            r = json.loads(f.read_text())
            out[r['id']] = r
    return out


def combined_scores(raters, ids):
    """Per proposal and metric: the mean of the raters' non-na values, or 'na'."""
    out = {}
    for pid in ids:
        out[pid] = {}
        for metric in METRICS:
            values = [r[pid]['scores'][metric] for r in raters.values()]
            numeric = [v for v in values if v != 'na']
            out[pid][metric] = sum(numeric) / len(numeric) if numeric else 'na'
    return out


def item_means(mapping, scores, metric):
    cells = {}
    for m in mapping:
        value = scores[m['id']][metric]
        if value != 'na':
            cells.setdefault((m['item'], m['arm']), []).append(value)
    return {cell: sum(v) / len(v) for cell, v in cells.items()}


CLUSTERS = None  # attempt003: item -> episode; differences are averaged within episode


def paired(means, items, arm, base='old', interval=False):
    pairs = {i: means[(i, arm)] - means[(i, base)]
             for i in items if (i, arm) in means and (i, base) in means}
    if not pairs:
        return None
    units = pairs
    if CLUSTERS:
        grouped = {}
        for i, d in pairs.items():
            grouped.setdefault(CLUSTERS[i], []).append(d)
        units = {c: sum(v) / len(v) for c, v in grouped.items()}
    diffs = list(units.values())
    out = {'items': len(diffs), 'estimate': round(sum(diffs) / len(diffs), 4),
           'p': round(signflip_p(diffs), 6),
           'per_item': {i: round(d, 4) for i, d in sorted(pairs.items())}}
    if CLUSTERS:
        out.update(unit='episode', n_items=len(pairs),
                   per_episode={c: round(d, 4) for c, d in sorted(units.items())})
    if interval:
        out['interval95'] = inversion_interval(diffs)
    return out


def clears(contrast):
    return contrast is not None and contrast['estimate'] >= MARGIN and contrast['p'] < ALPHA


def contrasts(mapping, scores, key):
    items_a = sorted(i for i, k in key.items() if k['type'] == 'A')
    items_all = sorted(key)
    items_null = sorted(i for i, k in key.items() if k['null'])
    means = {r: item_means(mapping, scores, r) for r in METRICS}
    out = {
        'primary': paired(means['R1'], items_a, 'new', interval=True),
        'directive_control': paired(means['R1'], items_a, 'directive', interval=True),
        'secondary': {
            'R2_new_minus_old_typeA': paired(means['R2'], items_a, 'new'),
            'R4_new_minus_old_all': paired(means['R4'], items_all, 'new'),
            'R5_new_minus_old_all': paired(means['R5'], items_all, 'new'),
            'R6_new_minus_old_null': paired(means['R6'], items_null, 'new'),
            'R2_directive_minus_old_typeA': paired(means['R2'], items_a, 'directive'),
        },
        'R3_failures': {arm: round(sum(1 - scores[m['id']]['R3'] for m in mapping if m['arm'] == arm), 2)
                        for arm in ARMS},
    }
    return out, means


def decide(c):
    s = c['secondary']
    harms = {
        'R3_failures_new_not_higher': c['R3_failures']['new'] <= c['R3_failures']['old'],
        'R4_mean_increase_at_most_0.5': s['R4_new_minus_old_all']['estimate'] <= R4_TOLERANCE,
        'R2_not_lower': s['R2_new_minus_old_typeA']['estimate'] >= 0,
    }
    surface_only = (not clears(c['primary']) and clears(s['R5_new_minus_old_all'])
                    and not clears(s['R2_new_minus_old_typeA']))
    if not clears(c['directive_control']):
        decision = 'underidentified'
    elif clears(c['primary']) and all(harms.values()):
        decision = 'retain'
    elif clears(c['primary']) or surface_only:
        decision = 'revise'
    else:
        decision = 'no_detectable_effect'
    return decision, harms, surface_only


def analyze(run, materials, results, figure_path, rater_dirs):
    key = {k['item']: k for k in json.loads((materials / 'key.json').read_text())}
    mapping = list(csv.DictReader(open(run / 'sealed' / 'mapping.csv', newline='')))
    key = {i: k for i, k in key.items() if i in {m['item'] for m in mapping}}  # items in this run
    ids = [m['id'] for m in mapping]
    raters = {d: load_ratings(run, d) for d in rater_dirs}
    for d, r in raters.items():
        absent = [pid for pid in ids if pid not in r]
        if absent:
            raise SystemExit(f'{d}: {len(absent)} proposals lack a rating')
    combined = combined_scores(raters, ids)
    main, means = contrasts(mapping, combined, key)
    decision, harms, surface_only = decide(main)
    by_rater = {}
    for d in rater_dirs:
        single = {pid: raters[d][pid]['scores'] for pid in ids}
        c, _ = contrasts(mapping, single, key)
        by_rater[d] = {'primary': c['primary'], 'directive_control': c['directive_control'],
                       'R3_failures': c['R3_failures'], 'decision': decide(c)[0]}
    agreement = None
    if len(rater_dirs) == 2:
        x, y = (raters[d] for d in rater_dirs)
        a_ids = [m['id'] for m in mapping if key[m['item']]['type'] == 'A']
        agreement = {'raters': list(rater_dirs),
                     'R1_typeA': kappa([x[p]['scores']['R1'] for p in a_ids], [y[p]['scores']['R1'] for p in a_ids]),
                     'R3_all': kappa([x[p]['scores']['R3'] for p in ids], [y[p]['scores']['R3'] for p in ids])}
    k1 = agreement['R1_typeA']['kappa'] if agreement else None
    new_failures = []
    for m in mapping:
        if m['arm'] == 'new' and combined[m['id']]['R3'] < 1:
            new_failures.append({'id': m['id'], 'item': m['item'],
                                 'R3_by_rater': {d: raters[d][m['id']]['scores']['R3'] for d in rater_dirs},
                                 'reasons': {d: raters[d][m['id']]['reasons'].get('R3', '') for d in rater_dirs}})
    report = dict(main)
    report.update({
        'raters': list(rater_dirs), 'agreement': agreement,
        'primary_scores': ('mean of the model raters; ' +
                           ('kappa for R1 meets the 0.6 rule' if k1 is not None and k1 >= KAPPA_MIN
                            else 'rater-dependent: kappa for R1 below 0.6 or undefined') +
                           '; not validated by human ratings'),
        'by_rater': by_rater, 'new_arm_R3_failures': new_failures, 'harms': harms,
        'surface_compliance_pattern': surface_only, 'decision': decision,
        'blinding_check': 'unavailable: the scientist waived the blind sample on 2026-10-16',
        'operationalization': {
            'clears': f'estimate >= {MARGIN} and exact two-sided p < {ALPHA}',
            'order': 'directive control first; then primary with harms; then the R5-only pattern',
            'R3_failures': 'sum over proposals of 1 minus the rater-mean R3',
            'R4': 'mean over all items of the New minus Old item-mean cost',
        },
        'counts': {'proposals': len(mapping), 'ratings': {d: len(r) for d, r in raters.items()}},
    })
    results.mkdir(parents=True, exist_ok=True)
    (results / 'analysis.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    with open(results / 'per_item.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'type', 'null', 'arm', 'draws'] + list(METRICS))
        for item in sorted(key):
            for arm in ARMS:
                draws = sum(m['item'] == item and m['arm'] == arm for m in mapping)
                if draws:
                    w.writerow([item, key[item]['type'], int(key[item]['null']), arm, draws]
                               + ['' if (item, arm) not in means[r] else round(means[r][(item, arm)], 4)
                                  for r in METRICS])
    items_a = sorted(i for i, k in key.items() if k['type'] == 'A')
    figure(figure_path, items_a, means['R1'], main['primary'], main['directive_control'])
    return report


def describe(run, materials, results, figure_path, rater_dirs):
    """Effect sizes only: no test, interval or decision (attempt003 gap items)."""
    key = {k['item']: k for k in json.loads((materials / 'key.json').read_text())}
    mapping = list(csv.DictReader(open(run / 'sealed' / 'mapping.csv', newline='')))
    key = {i: k for i, k in key.items() if i in {m['item'] for m in mapping}}
    ids = [m['id'] for m in mapping]
    raters = {d: load_ratings(run, d) for d in rater_dirs}
    for d, r in raters.items():
        absent = [pid for pid in ids if pid not in r]
        if absent:
            raise SystemExit(f'{d}: {len(absent)} proposals lack a rating')
    items = sorted(key)

    def effects(scores):
        means = {r: item_means(mapping, scores, r) for r in METRICS}
        out = {}
        for metric in METRICS:
            out[metric] = {}
            for arm in ('new', 'directive'):
                per_item = {i: round(means[metric][(i, arm)] - means[metric][(i, 'old')], 4)
                            for i in items if (i, arm) in means[metric] and (i, 'old') in means[metric]}
                if not per_item:
                    continue
                grouped = {}
                for i, d in per_item.items():
                    grouped.setdefault((CLUSTERS or {}).get(i, i), []).append(d)
                per_episode = {e: round(sum(v) / len(v), 4) for e, v in sorted(grouped.items())}
                out[metric][f'{arm}_minus_old'] = {
                    'mean_over_episodes': round(sum(per_episode.values()) / len(per_episode), 4),
                    'mean_over_items': round(sum(per_item.values()) / len(per_item), 4),
                    'episodes': len(per_episode), 'items': len(per_item),
                    'per_episode': per_episode, 'per_item': per_item}
        arm_means = {arm: round(sum(means['R1'][(i, arm)] for i in items) / len(items), 4) for arm in ARMS}
        return out, means, arm_means

    combined = combined_scores(raters, ids)
    eff, means, arm_means = effects(combined)
    by_rater = {d: effects({pid: raters[d][pid]['scores'] for pid in ids})[0]['R1'] for d in rater_dirs}
    report = {
        'note': 'Descriptive only (scientist decision 2026-10-17): no test, interval or decision.',
        'items': items, 'episodes': sorted({(CLUSTERS or {}).get(i, i) for i in items}),
        'R1_mean_by_arm_over_items': arm_means, 'effects': eff, 'R1_by_rater': by_rater,
        'R3_failures': {arm: round(sum(1 - combined[m['id']]['R3'] for m in mapping if m['arm'] == arm), 2)
                        for arm in ARMS},
        'counts': {'proposals': len(mapping), 'ratings': {d: len(r) for d, r in raters.items()}},
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / 'descriptive.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    with open(results / 'per_item.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'episode', 'null', 'arm', 'draws'] + list(METRICS))
        for item in items:
            for arm in ARMS:
                draws = sum(m['item'] == item and m['arm'] == arm for m in mapping)
                w.writerow([item, (CLUSTERS or {}).get(item, item), int(key[item]['null']), arm, draws]
                           + ['' if (item, arm) not in means[r] else round(means[r][(item, arm)], 4)
                              for r in METRICS])
    r1 = eff['R1']
    shown = {arm: {'estimate': r1[f'{arm}_minus_old']['mean_over_episodes'],
                   'items': r1[f'{arm}_minus_old']['episodes'], 'unit': 'episode'}
             for arm in ('new', 'directive')}
    figure(figure_path, items, means['R1'], shown['new'], shown['directive'])
    return report


def figure(path, items, r1, primary, control):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, text, secondary, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
    colors = {'old': '#2a78d6', 'new': '#eb6834', 'directive': '#1baf7a'}
    labels = {'old': 'Old instructions', 'new': 'Three result rules', 'directive': 'Directive control'}
    markers = {'old': 'o', 'new': 's', 'directive': '^'}
    offsets = {'old': 0.22, 'new': 0.0, 'directive': -0.22}
    plt.rcParams.update({'font.size': 9, 'text.color': text, 'axes.labelcolor': secondary,
                         'xtick.color': secondary, 'ytick.color': text})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, max(4.8, 0.36 * len(items) + 1.2)),
                                 facecolor=surface,
                                 gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.35})
    for a in (ax, bx):
        a.set_facecolor(surface)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(muted)
        a.set_axisbelow(True)
    n = len(items)
    for row, item in enumerate(items):
        y = n - 1 - row
        for arm in ARMS:
            if (item, arm) in r1:
                ax.scatter(r1[(item, arm)], y + offsets[arm], s=46, marker=markers[arm],
                           color=colors[arm], edgecolors=surface, linewidths=1.5, zorder=3,
                           label=labels[arm] if row == 0 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(items)))
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.set_xlim(-0.06, 1.06)
    ax.grid(axis='x', color=grid, linewidth=0.8)
    ax.set_xlabel('Share of draws whose proposal tests a named surviving\n'
                  'alternative (R1, mean of two model raters)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    rows = [('Rules − old', primary, colors['new'], markers['new']),
            ('Directive − old', control, colors['directive'], markers['directive'])]
    descriptive = 'interval95' not in primary
    for k, (label, c, color, marker) in enumerate(rows):
        y = 1 - k
        if descriptive:
            note = f"{c['estimate']:+.2f}\n(descriptive)"
        else:
            iv = c['interval95']
            bx.plot([iv['low'], iv['high']], [y, y], color=color, linewidth=2,
                    solid_capstyle='round', zorder=2)
            note = f"{c['estimate']:+.2f} [{iv['low']:+.2f}, {iv['high']:+.2f}]\np = {c['p']:.3g}"
        bx.scatter(c['estimate'], y, s=46, marker=marker, color=color, edgecolors=surface,
                   linewidths=1.5, zorder=3)
        bx.annotate(note, (c['estimate'], y), xytext=(0, 9), textcoords='offset points', ha='center',
                    va='bottom', fontsize=8.5, color=text, zorder=4,
                    bbox={'boxstyle': 'round,pad=0.2', 'fc': surface, 'ec': 'none'})
    bx.axvline(0, color=muted, linewidth=1, zorder=1)
    if not descriptive:
        bx.axvline(MARGIN, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        bx.text(MARGIN, -0.55, ' margin 0.25', color=secondary, fontsize=8, ha='left', va='bottom')
    bx.set_xlim(-1.05, 1.05)
    bx.set_ylim(-0.6, 1.75)
    bx.set_yticks([1, 0])
    bx.set_yticklabels([r[0] for r in rows])
    bx.grid(axis='x', color=grid, linewidth=0.8)
    units = 'episodes' if primary.get('unit') == 'episode' else 'type-A items'
    bx.set_xlabel(f'Mean paired difference over {primary["items"]} {units}\n'
                  + ('descriptive only: no interval or test' if descriptive else 'with 95% interval'))
    fig.savefig(path, dpi=200, facecolor=surface, bbox_inches='tight')
    plt.close(fig)


def selftest(scratch):
    assert signflip_p([1.0] * 10) == 2 / 1024
    assert signflip_p([0.0] * 5) == 1.0
    rng = random.Random(1)
    lattice = [k / 12 for k in range(-12, 13)]
    for _ in range(20):
        d = [rng.choice(lattice) for _ in range(8)]
        brute = sum(abs(sum(s * x for s, x in zip(p, d))) >= abs(sum(d)) - 1e-9
                    for p in itertools.product((1, -1), repeat=8)) / 256
        assert abs(brute - signflip_p(d)) < 1e-12, (d, brute)
    iv = inversion_interval([1.0] * 10)
    assert iv == {'low': 1.0, 'high': 1.0, 'contiguous': True}, iv
    d = [1, 2 / 3, 1, 1 / 3, 0, 1, 2 / 3, 1, 1 / 3, 2 / 3]
    iv = inversion_interval(d)
    assert iv['low'] <= sum(d) / len(d) <= iv['high'] and iv['contiguous'], iv
    assert kappa([0, 1, 0, 1], [0, 1, 0, 1])['kappa'] == 1.0
    assert kappa([0, 1, 0, 1], [1, 0, 1, 0])['kappa'] == -1.0
    raters = {'a': {'p1': {'scores': {m: 1 for m in METRICS}}}, 'b': {'p1': {'scores': {m: 0 for m in METRICS}}}}
    raters['b']['p1']['scores']['R5'] = 'na'
    c = combined_scores(raters, ['p1'])['p1']
    assert c['R1'] == 0.5 and c['R5'] == 1, c
    global CLUSTERS
    CLUSTERS = {'a1': 'E1', 'a2': 'E1', 'b1': 'E2'}
    m = {('a1', 'new'): 1.0, ('a1', 'old'): 0.0, ('a2', 'new'): 0.5, ('a2', 'old'): 0.0,
         ('b1', 'new'): 1 / 3, ('b1', 'old'): 0.0}
    c = paired(m, ['a1', 'a2', 'b1'], 'new')
    assert c['items'] == 2 and c['n_items'] == 3 and abs(c['estimate'] - (0.75 + 1 / 3) / 2) < 1e-3, c
    assert c['p'] == 0.5, c  # two positive episodes: 2 of 4 sign patterns are as extreme
    CLUSTERS = None
    items = [f'EP{i:02d}' for i in range(1, 11)]
    r1 = {(i, a): rng.choice([k / 6 for k in range(7)]) for i in items for a in ARMS}
    scratch.mkdir(parents=True, exist_ok=True)
    figure(scratch / 'selftest_figure.png', items, r1, paired(r1, items, 'new', interval=True),
           paired(r1, items, 'directive', interval=True))
    print('selftest passed; synthetic figure at', scratch / 'selftest_figure.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'describe', 'selftest'])
    ap.add_argument('--materials', help='outputs/arm_comparison')
    ap.add_argument('--run-dir', help='outputs/runs/arm_comparison/<attempt>')
    ap.add_argument('--raters', default='ratings,ratings_astra',
                    help='comma-separated rating directories inside the run')
    ap.add_argument('--results', help='released tables directory')
    ap.add_argument('--figure', help='released figure path (.png)')
    ap.add_argument('--scratch', help='selftest output directory')
    ap.add_argument('--clusters', help='JSON item -> episode map; analyze by episode (attempt003)')
    args = ap.parse_args()
    if args.mode == 'selftest':
        return selftest(Path(args.scratch))
    if args.clusters:
        global CLUSTERS
        CLUSTERS = json.loads(Path(args.clusters).read_text())
    if args.mode == 'describe':
        report = describe(Path(args.run_dir), Path(args.materials), Path(args.results),
                          Path(args.figure), args.raters.split(','))
        print(json.dumps({'R1_mean_by_arm_over_items': report['R1_mean_by_arm_over_items'],
                          'R1': {k: v['mean_over_episodes'] for k, v in report['effects']['R1'].items()}},
                         indent=1))
        return
    report = analyze(Path(args.run_dir), Path(args.materials), Path(args.results),
                     Path(args.figure), args.raters.split(','))
    print(json.dumps({k: report[k] for k in ('decision', 'primary_scores')}, indent=1))


if __name__ == '__main__':
    main()
