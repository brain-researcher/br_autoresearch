#!/usr/bin/env python3
"""EP24 attempt003: directive-screened items with testable alternatives.

Modes:
  extract  ask Codex (gpt-6-astra, max) for up to three resume-point entries per
           episode log at commit 960ab41, and to tag the alternatives named in
           each type-A README packet; every alternative is marked testable or
           claim_scope
  build    verify every quote against its source and write packets, key and
           episode clusters into the attempt003 materials directory
  retain   apply the screening rule: keep items where old-arm R1 <= 0.25 and
           directive R1 >= 0.75; stop unless >= 10 items in >= 7 episodes remain
"""
import argparse
import concurrent.futures as cf
import csv
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path('/oak/stanford/groups/russpold/users/zijiao/br_autoresearch')
EP24 = REPO / 'episode24_result_driven_harness'
COMMIT = '960ab41'
MAX_WORDS = 400
NEWEST = 60
PICKS = 3
OLD_MAX, DIRECTIVE_MIN = 0.25, 0.75
MIN_ITEMS, MIN_EPISODES = 10, 7

KINDS = """For each alternative set "kind":
- "testable": an explanation, confound or limitation that a next analysis on the episode's permitted
  data could weaken or rule out;
- "claim_scope": a statement only about what the result does not establish (for example "not a
  mechanism", "exploratory", "not held-out evidence"), which no next analysis on permitted data tests.
Quotes must be copied exactly from the source text, without ellipses. "closed_material" lists only
exact quotes saying that data, sets or analyses are closed, held out, sealed, reserved or unopened;
use [] when there are none."""

TASK_LOG = """You are selecting decision points from a research episode's execution log for an experiment
on research decisions. You cannot run commands or read files; answer from this text alone.

Episode: {ep} — {title}
Below are the episode's newest log entries, newest first. Each starts with its heading line.

Select up to THREE of the NEWEST entries that each meet all of these:
1. It was written while the episode was active: it does not close, stop or finalize the episode.
2. It reports a completed scientific result: an estimate, evaluation or comparison computed on the
   episode's scientific data, such as a model comparison or an effect estimate. Numerical or optimizer
   progress, validation or eligibility checks, feasibility or support assessments, source-metadata or
   provenance checks, power projections, submissions, accounting, figures and documentation do not count.
3. Its own text names at least one surviving alternative to the favored reading of that result: an
   explanation, confound or limitation that the result does not rule out.
4. Its own text states a next action.
(Every entry below already has at most 400 words.)

{kinds}

Return one JSON object and nothing else, picks ordered newest first:
{{"picks": [{{"heading": "<heading line exactly, without the leading ##>", "reason": "<one sentence>",
  "result_quote": "<exact quote stating the result>",
  "alternatives": [{{"quote": "<exact quote>", "gloss": "<one-line reading>", "kind": "testable or claim_scope"}}],
  "next_action_quote": "<exact quote>", "plan_tests_alternative": true or false,
  "null_result": true or false, "closed_material": ["<exact quote>"]}}]}}
"plan_tests_alternative" is true only if the logged next action tests a testable alternative.
Return {{"picks": []}} if no entry qualifies.

=== Log entries ===

{entries}
"""

TASK_README = """You are preparing a scoring key for one research decision point. You cannot run commands
or read files; answer from the summary text alone.

Below is the summary an agent sees for episode {ep}. Return the exact quote stating the current
result, every surviving alternative explanation, confound or limitation of that result that the
summary names, the closed material, and whether the result is a null, a failed margin or an equivalence.

{kinds}

Return one JSON object and nothing else:
{{"result_quote": "<exact quote>",
  "alternatives": [{{"quote": "<exact quote>", "gloss": "<one-line reading>", "kind": "testable or claim_scope"}}],
  "null_result": true or false, "closed_material": ["<exact quote>"]}}

=== Summary ===

{packet}
"""


def episode_dirs():
    for d in sorted(REPO.glob('episode[0-2][0-9]_*')):
        if d.name[7:9] != '24':
            yield d.name[7:9], d


def git_show(path):
    # Sherlock's git 1.8.3 has no `git -C`; run from the repository instead.
    return subprocess.run(['git', 'show', f'{COMMIT}:{path}'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def entries(log_text):
    """Dated entries as (heading, body, date), newest first."""
    out, cur = [], None
    for line in log_text.splitlines():
        if line.startswith('## '):
            if cur:
                out.append(cur)
            cur = [line[3:].strip(), []]
        elif cur is not None:
            cur[1].append(line)
    if cur:
        out.append(cur)
    dated = []
    for i, (heading, body) in enumerate(out):
        m = re.match(r'(\d{4}-\d{2}-\d{2})', heading)
        if m:
            dated.append((m.group(1), i, heading, '\n'.join(body).strip()))
    if dated and dated[0][0] < dated[-1][0]:
        dated.sort(key=lambda e: (e[0], e[1]), reverse=True)
    else:
        dated.sort(key=lambda e: (e[0], -e[1]), reverse=True)
    return [(h, b, d) for d, _, h, b in dated]


def clean(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text.replace('**', '')


def norm(text):
    return re.sub(r'\s+', ' ', clean(text)).strip().strip('"“”')


def words(text):
    return len(clean(text).split())


def title_of(d):
    title = re.sub(r'^#\s*', '', git_show(f'{d.name}/GOAL.md').splitlines()[0])
    return re.sub(r'^EP\d\d:\s*', '', title)


def codex(prompt, scratch, tag, timeout=1800):
    cwd = scratch / 'a3_extract_cwd' / tag
    cwd.mkdir(parents=True, exist_ok=True)
    out = scratch / 'a3_extract_cwd' / f'{tag}.txt'
    cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
           '--color', 'never', '-s', 'read-only', '-m', 'gpt-6-astra',
           '-c', 'model_reasoning_effort="max"', '-c', 'approval_policy="never"',
           '-C', str(cwd), '-o', str(out), '-']
    r = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
    text = out.read_text() if out.exists() else ''
    m = re.search(r'\{.*\}', text, re.S)
    return (json.loads(m.group(0)) if r.returncode == 0 and m else None), r.stderr[-800:]


def extract(run, scratch):
    out_dir = run / 'extraction'
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = {p['item']: p['packet'] for p in
              json.loads((EP24 / 'outputs/arm_comparison/packets.json').read_text())}
    type_a = [k['item'] for k in json.loads((EP24 / 'outputs/arm_comparison/key.json').read_text())
              if k['type'] == 'A']
    tasks = []
    for ep, d in episode_dirs():
        tasks.append(('log', ep, d))
        if f'EP{ep}' in type_a:
            tasks.append(('readme', ep, d))

    def one(kind, ep, d):
        final = out_dir / f'EP{ep}_{kind}.json'
        if final.exists():
            return f'EP{ep} {kind}', 'skipped'
        if kind == 'log':
            ents = [e for e in entries(git_show(f'{d.name}/outputs/experiment_log.md'))
                    if words(e[0] + '\n' + e[1]) <= MAX_WORDS][:NEWEST]
            prompt = TASK_LOG.format(ep=f'EP{ep}', title=title_of(d), kinds=KINDS,
                                     entries='\n\n'.join(f'## {h}\n{b}' for h, b, _ in ents))
        else:
            prompt = TASK_README.format(ep=f'EP{ep}', kinds=KINDS, packet=readme[f'EP{ep}'])
        for attempt in (1, 2):
            try:
                obj, err = codex(prompt, scratch, f'EP{ep}_{kind}_{attempt}')
            except json.JSONDecodeError:
                obj, err = None, 'reply was not valid JSON'
            if obj is not None:
                obj.update(episode=f'EP{ep}', kind=kind)
                final.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
                return f'EP{ep} {kind}', 'ok'
        (out_dir / f'EP{ep}_{kind}.failed.txt').write_text(err + '\n')
        return f'EP{ep} {kind}', 'failed'

    with cf.ThreadPoolExecutor(4) as pool:
        for fut in cf.as_completed([pool.submit(one, *t) for t in tasks]):
            label, status = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] extract {label} {status}", flush=True)


def check_quotes(pick, source_text, need_next):
    text = norm(source_text)
    quotes = [pick.get('result_quote', '')] + [a.get('quote', '') for a in pick.get('alternatives', [])]
    quotes += list(pick.get('closed_material', []))
    if need_next:
        quotes.append(pick.get('next_action_quote', ''))
    bad = [q for q in quotes if not q or '...' in q or '…' in q or norm(q) not in text]
    kinds = {a.get('kind') for a in pick.get('alternatives', [])}
    problems = [f'{len(bad)} quotes not verbatim: {bad[:2]}'] if bad else []
    if not kinds <= {'testable', 'claim_scope'}:
        problems.append(f'unknown kinds {kinds}')
    if not any(a.get('kind') == 'testable' for a in pick.get('alternatives', [])):
        problems.append('no testable alternative')
    return problems


def key_entry(item, episode, pick, logged):
    testable = [a for a in pick['alternatives'] if a['kind'] == 'testable']
    scope = [a for a in pick['alternatives'] if a['kind'] == 'claim_scope']
    lines = [f'### {item} (episode {episode})' + (' — null' if pick.get('null_result') else ''),
             f"- Current reading: \"{norm(pick['result_quote'])}\"", '- Surviving alternatives:']
    lines += [f"  {i}. \"{norm(a['quote'])}\". {a.get('gloss', '').strip()}" for i, a in enumerate(testable, 1)]
    if scope:
        lines.append('- Claim limits named in the record (not alternatives; never count for R1): '
                     + '; '.join(f'"{norm(a["quote"])}"' for a in scope))
    if logged:
        lines.append(f"- Logged next action: \"{norm(pick['next_action_quote'])}\" (tests a listed "
                     f"alternative: {'yes' if pick.get('plan_tests_alternative') else 'no'})")
    closed = [f'"{norm(q)}"' for q in pick.get('closed_material', [])]
    lines.append('- Closed material: ' + ('; '.join(closed) if closed else 'none stated.'))
    return '\n'.join(lines)


def build(run, materials):
    materials.mkdir(parents=True, exist_ok=True)
    readme = {p['item']: p['packet'] for p in
              json.loads((EP24 / 'outputs/arm_comparison/packets.json').read_text())}
    packets, key, clusters, report = [], [], {}, []
    for ep, d in episode_dirs():
        episode = f'EP{ep}'
        rpath = run / 'extraction' / f'{episode}_readme.json'
        if rpath.exists():
            pick = json.loads(rpath.read_text())
            problems = check_quotes(pick, readme[episode], need_next=False)
            item = f'{episode}-R'
            if problems:
                report.append({'item': item, 'status': 'excluded', 'problems': problems})
            else:
                packets.append({'item': item, 'episode': episode, 'packet': readme[episode]})
                key.append({'item': item, 'type': 'A', 'null': bool(pick.get('null_result')),
                            'chain': False, 'closed': False, 'entry': key_entry(item, episode, pick, False),
                            'source': f'README.md@{COMMIT} navigation row'})
                clusters[item] = episode
                report.append({'item': item, 'status': 'candidate'})
        lpath = run / 'extraction' / f'{episode}_log.json'
        if not lpath.exists():
            report.append({'item': f'{episode}-L', 'status': 'no extraction'})
            continue
        ents = {h: (b, dt) for h, b, dt in entries(git_show(f'{d.name}/outputs/experiment_log.md'))}
        title = title_of(d)
        seen = set()
        for n, pick in enumerate(json.loads(lpath.read_text()).get('picks', [])[:PICKS], 1):
            item = f'{episode}-L{n}'
            heading = pick.get('heading', '').strip().lstrip('#').strip()
            if heading not in ents or heading in seen:
                report.append({'item': item, 'status': 'excluded', 'problems': ['heading not found or repeated']})
                continue
            seen.add(heading)
            body, date = ents[heading]
            problems = check_quotes(pick, heading + '\n' + body, need_next=True)
            if words(heading + '\n' + body) > MAX_WORDS:
                problems.append('over 400 words')
            if date > '2026-10-15':
                problems.append(f'dated {date}')
            if problems:
                report.append({'item': item, 'status': 'excluded', 'problems': problems, 'heading': heading})
                continue
            entry = clean(f'{heading}\n{body}').strip()
            packets.append({'item': item, 'episode': episode,
                            'packet': f'{episode} — {title}\nCurrent log entry ({date}):\n{entry}'})
            key.append({'item': item, 'type': 'A', 'null': bool(pick.get('null_result')),
                        'chain': False, 'closed': False, 'entry': key_entry(item, episode, pick, True),
                        'plan_tests_alternative': bool(pick.get('plan_tests_alternative')),
                        'source': f'{d.name}/outputs/experiment_log.md@{COMMIT} — {heading}'})
            clusters[item] = episode
            report.append({'item': item, 'status': 'candidate', 'date': date})
    (materials / 'packets.json').write_text(json.dumps(packets, indent=1, ensure_ascii=False) + '\n')
    (materials / 'key.json').write_text(json.dumps(key, indent=1, ensure_ascii=False) + '\n')
    (materials / 'clusters.json').write_text(json.dumps(clusters, indent=1) + '\n')
    (materials / 'extraction_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    md = ['# EP24 attempt003 candidate packets', '',
          f'README rows and log entries at `{COMMIT}`. Codex (gpt-6-astra, max) drafted the picks and the '
          'testable/claim-scope tags; every quote was verified verbatim against its source. Proposers see '
          'the packet only; raters also see [key.md](key.md).', '']
    for p in packets:
        md += [f"## {p['item']}", '', '~~~~text', p['packet'], '~~~~', '']
    (materials / 'items.md').write_text('\n'.join(md))
    (materials / 'key.md').write_text('# EP24 attempt003 key\n\n' + '\n\n'.join(k['entry'] for k in key) + '\n')
    for r in report:
        print(json.dumps(r, ensure_ascii=False))
    print(f'candidates: {len(packets)} items in {len(set(clusters.values()))} episodes')


def retain(screen_run, materials, rater_dirs):
    mapping = list(csv.DictReader(open(screen_run / 'sealed' / 'mapping.csv', newline='')))
    values = {}
    for m in mapping:
        for d in rater_dirs:
            r = json.loads((screen_run / d / f"{m['id']}.json").read_text())
            values.setdefault((m['item'], m['arm']), []).append(r['scores']['R1'])
    clusters = json.loads((materials / 'clusters.json').read_text())
    items = sorted({i for i, _ in values})
    means = {i: {arm: round(sum(values[(i, arm)]) / len(values[(i, arm)]), 4)
                 for arm in ('old', 'directive') if (i, arm) in values} for i in items}
    retained = [i for i in items if means[i].get('old', 1) <= OLD_MAX and means[i].get('directive', 0) >= DIRECTIVE_MIN]
    episodes = sorted({clusters[i] for i in retained})
    out = {'rule': f'keep when rater-mean old R1 <= {OLD_MAX} and directive R1 >= {DIRECTIVE_MIN}',
           'screen_means': means, 'retained': retained, 'retained_episodes': episodes,
           'dropped': [i for i in items if i not in retained]}
    (materials / 'retained.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    if len(retained) < MIN_ITEMS or len(episodes) < MIN_EPISODES:
        raise SystemExit(f'{len(retained)} items in {len(episodes)} episodes retained '
                         f'(need {MIN_ITEMS} in {MIN_EPISODES}); attempt003 stops')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['extract', 'build', 'retain'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--materials', required=True)
    ap.add_argument('--scratch')
    ap.add_argument('--raters', default='ratings,ratings_astra')
    args = ap.parse_args()
    run, materials = Path(args.run_dir), Path(args.materials)
    if args.mode == 'extract':
        extract(run, Path(args.scratch))
    elif args.mode == 'build':
        build(run, materials)
    else:
        retain(run, materials, args.raters.split(','))


if __name__ == '__main__':
    main()
