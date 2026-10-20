#!/usr/bin/env python3
"""EP24 attempt002: resume-point items from episode logs at commit 960ab41.

Modes:
  extract  ask Codex (gpt-6-astra, max) to pick each episode's newest qualifying
           log entry and quote it; one text-only call per episode
  build    verify every pick against the committed log, then write packets,
           key and readable copies into the attempt002 materials directory
  retain   apply the screening rule to the screening run's ratings

A qualifying entry was written while the episode was active, reports a
scientific result, names a surviving alternative or limitation, states a next
action and has at most 300 words (GOAL.md, attempt002).
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
COMMIT = '960ab41'
MAX_WORDS = 300
NEWEST = 60
DROP_AT = 0.75
# Lowered from 8 by the scientist on 2026-10-17, after screening and before any
# new- or directive-arm proposal existed.
MIN_RETAINED = 7

TASK = """You are selecting one decision point from a research episode's execution log for an \
experiment on research decisions. You cannot run commands or read files; answer from this text alone.

Episode: {ep} — {title}
Below are the episode's newest log entries, newest first. Each starts with its heading line.

Select the NEWEST entry that meets all of these:
1. It was written while the episode was active: it does not close, stop or finalize the episode.
2. It reports a completed scientific result: an estimate, evaluation or comparison computed on the
   episode's scientific data, such as a model comparison or an effect estimate. Numerical or optimizer
   progress, validation or eligibility checks, feasibility or support assessments, source-metadata or
   provenance checks, power projections, submissions, accounting, figures and documentation do not count.
3. Its own text names at least one surviving alternative to the favored reading of that result: an
   explanation, confound or limitation that the result does not rule out.
4. Its own text states a next action.
(Every entry below already has at most 300 words.)

Quotes must be copied exactly from the selected entry's text, without ellipses.
"closed_material" lists only exact quotes saying that data, sets or analyses are closed, held out,
sealed, reserved or unopened; use [] when there are none.
Return one JSON object and nothing else:
{{"qualifies": true, "heading": "<selected heading line, exactly, without the leading ##>",
 "reason": "<one sentence>", "result_quote": "<exact quote stating the result>",
 "alternatives": [{{"quote": "<exact quote>", "gloss": "<one-line reading of the alternative>"}}],
 "next_action_quote": "<exact quote>", "plan_tests_alternative": true or false,
 "null_result": true or false, "closed_material": ["<exact quote>"]}}
If no entry qualifies, return {{"qualifies": false, "reason": "<one sentence>"}}.

=== Log entries ===

{entries}
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
    if dated and dated[0][0] < dated[-1][0]:  # chronological file: newest is last
        dated.sort(key=lambda e: (e[0], e[1]), reverse=True)
    else:
        dated.sort(key=lambda e: (e[0], -e[1]), reverse=True)
    return [(h, b, d) for d, _, h, b in dated]


def clean(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text.replace('**', '')


def norm(text):
    return re.sub(r'\s+', ' ', clean(text)).strip().strip('"“”')


def codex(prompt, cwd, out, timeout=1800):
    cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
           '--color', 'never', '-s', 'read-only', '-m', 'gpt-6-astra',
           '-c', 'model_reasoning_effort="max"', '-c', 'approval_policy="never"',
           '-C', str(cwd), '-o', str(out), '-']
    r = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
    return r.returncode, out.read_text() if out.exists() else '', r.stderr[-800:]


def extract(run, scratch):
    out_dir = run / 'extraction'
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(ep, d):
        final = out_dir / f'EP{ep}.json'
        if final.exists():
            return ep, 'skipped'
        title = re.sub(r'^#\s*', '', git_show(f'{d.name}/GOAL.md').splitlines()[0])
        title = re.sub(r'^EP\d\d:\s*', '', title)
        ents = [e for e in entries(git_show(f'{d.name}/outputs/experiment_log.md'))
                if len(clean(e[0] + '\n' + e[1]).split()) <= MAX_WORDS][:NEWEST]
        block = '\n\n'.join(f'## {h}\n{b}' for h, b, _ in ents)
        prompt = TASK.format(ep=f'EP{ep}', title=title, entries=block)
        for attempt in (1, 2):
            cwd = scratch / 'extract_cwd' / f'EP{ep}_{attempt}'
            cwd.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            rc, text, err = codex(prompt, cwd, scratch / 'extract_cwd' / f'EP{ep}_{attempt}.txt')
            m = re.search(r'\{.*\}', text, re.S)
            if rc == 0 and m:
                try:
                    pick = json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
                pick.update(item=f'EP{ep}', title=title, seconds=round(time.time() - t0, 1))
                final.write_text(json.dumps(pick, indent=1, ensure_ascii=False) + '\n')
                return ep, 'ok'
        (out_dir / f'EP{ep}.failed.txt').write_text(f'returncode {rc}\n{err}\n{text[:2000]}\n')
        return ep, 'failed'

    with cf.ThreadPoolExecutor(4) as pool:
        for fut in cf.as_completed([pool.submit(one, ep, d) for ep, d in episode_dirs()]):
            ep, status = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] extract EP{ep} {status}", flush=True)


def build(run, materials):
    materials.mkdir(parents=True, exist_ok=True)
    report, packets, key = [], [], []
    for ep, d in episode_dirs():
        path = run / 'extraction' / f'EP{ep}.json'
        if not path.exists():
            report.append({'item': f'EP{ep}', 'status': 'no extraction'})
            continue
        pick = json.loads(path.read_text())
        if not pick.get('qualifies'):
            report.append({'item': f'EP{ep}', 'status': 'none qualifies', 'reason': pick.get('reason')})
            continue
        ents = {h: (b, dt) for h, b, dt in entries(git_show(f'{d.name}/outputs/experiment_log.md'))}
        heading = pick['heading'].strip().lstrip('#').strip()
        problems = []
        if heading not in ents:
            report.append({'item': f'EP{ep}', 'status': 'heading not found', 'heading': heading})
            continue
        body, date = ents[heading]
        text = norm(heading + '\n' + body)
        words = len(clean(heading + '\n' + body).split())
        if words > MAX_WORDS:
            problems.append(f'{words} words')
        if date > '2026-10-15':
            problems.append(f'dated {date}')
        quotes = ([pick.get('result_quote', ''), pick.get('next_action_quote', '')]
                  + [a.get('quote', '') for a in pick.get('alternatives', [])]
                  + list(pick.get('closed_material', [])))
        missing = [q for q in quotes if not q or '...' in q or '…' in q or norm(q) not in text]
        if missing:
            problems.append(f'{len(missing)} quotes not verbatim: {missing[:2]}')
        if not pick.get('alternatives'):
            problems.append('no alternative quoted')
        if problems:
            report.append({'item': f'EP{ep}', 'status': 'failed verification', 'problems': problems,
                           'heading': heading})
            continue
        entry = clean(f'{heading}\n{body}').strip()
        packet = f"EP{ep} — {pick['title']}\nCurrent log entry ({date}):\n{entry}"
        lines = [f"### EP{ep}" + (' — null' if pick.get('null_result') else ''),
                 f"- Current reading: \"{norm(pick['result_quote'])}\"", '- Surviving alternatives:']
        lines += [f"  {i}. \"{norm(a['quote'])}\". {a.get('gloss', '').strip()}"
                  for i, a in enumerate(pick['alternatives'], 1)]
        lines.append(f"- Logged next action: \"{norm(pick['next_action_quote'])}\" (tests a listed "
                     f"alternative: {'yes' if pick.get('plan_tests_alternative') else 'no'})")
        closed = [f'"{norm(q)}"' for q in pick.get('closed_material', [])]
        lines.append('- Closed material: ' + ('; '.join(closed) if closed else 'none stated.'))
        packets.append({'item': f'EP{ep}', 'packet': packet})
        key.append({'item': f'EP{ep}', 'type': 'A', 'null': bool(pick.get('null_result')),
                    'chain': False, 'closed': False, 'entry': '\n'.join(lines),
                    'plan_tests_alternative': bool(pick.get('plan_tests_alternative')),
                    'source': f'{d.name}/outputs/experiment_log.md@{COMMIT} — {heading}', 'words': words})
        report.append({'item': f'EP{ep}', 'status': 'candidate', 'date': date, 'words': words,
                       'plan_tests_alternative': bool(pick.get('plan_tests_alternative'))})
    (materials / 'packets.json').write_text(json.dumps(packets, indent=1, ensure_ascii=False) + '\n')
    (materials / 'key.json').write_text(json.dumps(key, indent=1, ensure_ascii=False) + '\n')
    (materials / 'extraction_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    md = ['# EP24 attempt002 candidate packets', '',
          f'Built from each episode log at `{COMMIT}`; picks drafted by Codex (gpt-6-astra, max) and '
          'verified mechanically (heading present, at most 300 words, every quote verbatim). '
          'Proposers see the packet only; raters also see [key.md](key.md).', '']
    for p in packets:
        md += [f"## {p['item']}", '', '~~~~text', p['packet'], '~~~~', '']
    (materials / 'items.md').write_text('\n'.join(md))
    (materials / 'key.md').write_text('# EP24 attempt002 key\n\n' + '\n\n'.join(k['entry'] for k in key) + '\n')
    for r in report:
        print(json.dumps(r, ensure_ascii=False))
    print(f'candidates: {len(packets)}')


def retain(screen_run, materials, rater_dirs):
    mapping = list(csv.DictReader(open(screen_run / 'sealed' / 'mapping.csv', newline='')))
    values = {}
    for m in mapping:
        for d in rater_dirs:
            r = json.loads((screen_run / d / f"{m['id']}.json").read_text())
            values.setdefault(m['item'], []).append(r['scores']['R1'])
    means = {item: sum(v) / len(v) for item, v in sorted(values.items())}
    retained = [i for i, v in means.items() if v < DROP_AT]
    out = {'rule': f'drop when rater-mean old-arm R1 >= {DROP_AT} over the screening draws',
           'screen_means': means, 'retained': retained,
           'dropped': [i for i in means if i not in retained]}
    (materials / 'retained.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    if len(retained) < MIN_RETAINED:
        raise SystemExit(f'only {len(retained)} items retained (< {MIN_RETAINED}); attempt002 stops')


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
