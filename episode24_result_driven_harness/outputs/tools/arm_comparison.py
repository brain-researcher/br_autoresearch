#!/usr/bin/env python3
"""EP24 arm comparison: generate next-action proposals, then score them blind.

Modes:
  test      two proposals (old, new) on the non-item packet, then two ratings
  generate  every planned proposal; finished IDs are skipped on a rerun
  rate      model-rater scores for every finished proposal, in shuffled order
  sample    the blind scientist sample and its score sheet
  all       generate, rate, then sample

Proposers and the rater receive only the frozen materials inserted in their
prompts. Logs, proposal metadata and ratings never record the arm; the
id-to-arm mapping is written once to sealed/mapping.csv for the analysis.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fill(template, **values):
    # One pass, so inserted text is never rescanned for placeholders.
    return re.sub(r'<<([A-Z_]+)>>', lambda m: values[m.group(1)], template)


class Materials:
    def __init__(self, root, plan_name='plan.json'):
        self.root = Path(root)
        self.plan = json.loads((self.root / plan_name).read_text())
        packets = json.loads((self.root / 'packets.json').read_text())
        self.packets = {p['item']: p['packet'] for p in packets}
        # attempt003 item IDs (EP07-L1) differ from the episode the proposer is told it owns
        self.episode = {p['item']: p.get('episode', p['item']) for p in packets}
        self.key = {k['item']: k for k in json.loads((self.root / 'key.json').read_text())}
        if self.plan.get('items_file'):  # attempt002 stage 2: retained items only
            keep = set(json.loads((self.root / self.plan['items_file']).read_text())['retained'])
            self.key = {i: k for i, k in self.key.items() if i in keep}
        self.instructions = {arm: (self.root / rel).read_text().rstrip()
                             for arm, rel in self.plan['arms'].items()}
        second = self.root / self.plan.get('second_rater_file', 'second_rater.json')
        self.second_rater = json.loads(second.read_text()) if second.exists() else None
        prompts = self.root / self.plan.get('prompts_dir', 'prompts')
        self.proposer = (prompts / 'proposer.txt').read_text()
        self.directive = (prompts / 'directive.txt').read_text().strip()
        self.rater_system = (prompts / 'rater_system.txt').read_text().strip()
        self.rater_user = (prompts / 'rater_user.txt').read_text()

    def proposer_prompt(self, item, arm, packet=None):
        return fill(self.proposer, ITEM=self.episode.get(item, item), INSTRUCTIONS=self.instructions[arm],
                    PACKET=packet or self.packets[item],
                    DIRECTIVE=' ' + self.directive if arm == 'directive' else '')

    def rater_prompt(self, keyrec, packet, proposal):
        return fill(self.rater_user, PACKET=packet, TYPE=keyrec['type'],
                    NULL='yes' if keyrec['null'] else 'no', KEY_ENTRY=keyrec['entry'],
                    PROPOSAL=proposal)


def schedule(mat):
    """All (item, arm, draw) jobs in a seeded random order with opaque IDs."""
    plan = mat.plan
    rng = random.Random(plan['seed'])
    jobs = []
    run_arms = plan.get('run_arms', ['old', 'new', 'directive'])
    for item in sorted(mat.key):
        arms = [a for a in ('old', 'new') if a in run_arms]
        if mat.key[item]['type'] in plan['directive_types'] and 'directive' in run_arms:
            arms.append('directive')
        for arm in arms:
            for draw in range(1, plan['draws'] + 1):
                jobs.append({'item': item, 'arm': arm, 'draw': draw})
    rng.shuffle(jobs)
    used = set()
    for order, job in enumerate(jobs, 1):
        pid = 'p%06x' % rng.randrange(16 ** 6)
        while pid in used:
            pid = 'p%06x' % rng.randrange(16 ** 6)
        used.add(pid)
        job.update(id=pid, order=order)
    return jobs


def write_mapping(run, jobs):
    sealed = run / 'sealed'
    sealed.mkdir(exist_ok=True)
    path = sealed / 'mapping.csv'
    rows = [[j['id'], j['order'], j['item'], j['arm'], j['draw']] for j in jobs]
    if path.exists():
        with open(path, newline='') as f:
            old = [r for r in csv.reader(f)][1:]
        if old != [[str(x) for x in r] for r in rows]:
            raise SystemExit('sealed/mapping.csv differs from the frozen schedule')
        return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'order', 'item', 'arm', 'draw'])
        w.writerows(rows)


def codex_header(text):
    fields = {}
    for line in text.splitlines()[:60]:
        m = re.match(r'^(model|provider|approval|sandbox|reasoning effort): (.+)$', line.strip())
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def generate_one(mat, outdir, pid, item, arm, scratch, packet=None):
    final = outdir / f'{pid}.txt'
    if final.exists() and final.read_text().strip():
        return pid, 'skipped'
    p = mat.plan['proposer']
    prompt = mat.proposer_prompt(item, arm, packet)
    attempts = []
    for attempt in range(1, p['attempts'] + 1):
        cwd = scratch / 'codex_cwd' / f'{pid}_{attempt}'
        cwd.mkdir(parents=True, exist_ok=True)
        part = outdir / f'{pid}.part'
        cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
               '--color', 'never', '-s', 'read-only', '-m', p['model'],
               '-c', f'model_reasoning_effort="{p["reasoning_effort"]}"',
               '-c', 'approval_policy="never"', '-C', str(cwd), '-o', str(part), '-']
        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                               timeout=p['timeout_s'])
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = 'timeout', '', ''
        text = part.read_text().strip() if part.exists() else ''
        attempts.append({'attempt': attempt, 'returncode': rc,
                         'seconds': round(time.time() - t0, 1), 'chars': len(text),
                         'header': codex_header(out + '\n' + err),
                         'stderr_tail': err[-1500:] if rc != 0 or not text else ''})
        if rc == 0 and text:
            final.write_text(text + '\n')
            part.unlink()
            break
    meta = {'id': pid, 'words': len(final.read_text().split()) if final.exists() else 0,
            'attempts': attempts, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (outdir / f'{pid}.json').write_text(json.dumps(meta, indent=1) + '\n')
    return pid, 'ok' if final.exists() else 'failed'


SCORE_VALUES = {'R3': (0, 1), 'R4': (0, 1, 2, 3), 'R5': (0, 1, 'na')}


def normalize(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    if isinstance(value, str):
        return value.strip().lower()
    return value


def parse_scores(text, keyrec):
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError('no JSON object in rater reply')
    obj = json.loads(match.group(0))
    scores = {name: normalize(obj.get(name)) for name in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')}
    allowed = dict(SCORE_VALUES)
    allowed['R1'] = allowed['R2'] = (0, 1) if keyrec['type'] == 'A' else ('na',)
    allowed['R6'] = (0, 1) if keyrec['null'] else ('na',)
    bad = {n: v for n, v in scores.items() if v not in allowed[n]}
    if bad:
        raise ValueError(f'invalid scores {bad}')
    return scores, obj.get('unlisted_alternative'), obj.get('reasons', {})


# Stop rating after this many proposals fail in a row (a login or usage-limit
# error fails every call), instead of exhausting every proposal's attempts.
BREAKER = {'failures': 0, 'limit': 6, 'tripped': False}
RETRY_WAIT_S = 30


def call_rater(mat, rater, prompt, scratch, tag):
    """One rater call; returns (reply text, call metadata, stderr)."""
    if rater == 'claude':
        cfg = mat.plan['rater']
        cwd = scratch / 'rater_cwd'
        cwd.mkdir(parents=True, exist_ok=True)
        cmd = [cfg['bin'], '-p', '--model', cfg['model'], '--effort', cfg['effort'],
               '--tools', '', '--strict-mcp-config', '--no-session-persistence',
               '--output-format', 'json', '--system-prompt', mat.rater_system]
        r = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                           timeout=cfg['timeout_s'], cwd=cwd)
        reply = json.loads(r.stdout) if r.stdout.strip() else {}
        info = {'returncode': r.returncode, 'models': sorted(reply.get('modelUsage', {})),
                'turns': reply.get('num_turns'), 'subtype': reply.get('subtype'),
                'is_error': reply.get('is_error')}
        return reply.get('result') or '', info, r.stderr
    # Second model rater: the same rater prompt through codex exec, text only.
    cfg = mat.second_rater
    cwd = scratch / 'astra_rater_cwd' / tag
    cwd.mkdir(parents=True, exist_ok=True)
    out = scratch / 'astra_rater_out' / f'{tag}.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
           '--color', 'never', '-s', 'read-only', '-m', cfg['model'],
           '-c', f'model_reasoning_effort="{cfg["reasoning_effort"]}"',
           '-c', 'approval_policy="never"', '-C', str(cwd), '-o', str(out), '-']
    r = subprocess.run(cmd, input=mat.rater_system + '\n\n' + prompt, text=True,
                       capture_output=True, timeout=cfg['timeout_s'])
    text = out.read_text() if out.exists() else ''
    return text, {'returncode': r.returncode, 'header': codex_header(r.stdout + '\n' + r.stderr)}, r.stderr


def rate_one(mat, outdir, pid, keyrec, packet, proposal_path, scratch, rater='claude'):
    final = outdir / f'{pid}.json'
    if final.exists():
        return pid, 'skipped'
    if BREAKER['tripped']:
        return pid, 'stopped'
    r_cfg = mat.plan['rater'] if rater == 'claude' else mat.second_rater
    prompt = mat.rater_prompt(keyrec, packet, proposal_path.read_text().strip())
    attempts = []
    for attempt in range(1, r_cfg['attempts'] + 1):
        if attempt > 1:
            time.sleep(RETRY_WAIT_S * (attempt - 1))
        t0 = time.time()
        text, info, err, note = '', {}, '', ''
        try:
            text, info, err = call_rater(mat, rater, prompt, scratch, f'{pid}_{attempt}')
            scores, unlisted, reasons = parse_scores(text, keyrec)
            attempts.append(dict(info, attempt=attempt, seconds=round(time.time() - t0, 1)))
            record = {'id': pid, 'item': keyrec['item'], 'rater': rater, 'scores': scores,
                      'unlisted_alternative': unlisted, 'reasons': reasons, 'attempts': attempts,
                      'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
            final.write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n')
            BREAKER['failures'] = 0
            return pid, 'ok'
        except subprocess.TimeoutExpired:
            note = 'timeout'
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            note = f'{type(exc).__name__}: {exc}'[:400] + f' | {info} | {text[:400]}'
            if err.strip():
                note += f' | stderr {err[-400:]}'
        attempts.append({'attempt': attempt, 'seconds': round(time.time() - t0, 1), 'error': note})
    tag = os.environ.get('SLURM_JOB_ID', 'local')
    (outdir / f'{pid}.failed.{tag}.json').write_text(
        json.dumps({'id': pid, 'attempts': attempts}, indent=1) + '\n')
    BREAKER['failures'] += 1
    if BREAKER['failures'] >= BREAKER['limit']:
        BREAKER['tripped'] = True
        log(f'rating stopped after {BREAKER["failures"]} consecutive failed proposals')
    return pid, 'failed'


def run_pool(label, workers, tasks):
    done = collections.Counter()
    with cf.ThreadPoolExecutor(workers) as pool:
        for future in cf.as_completed([pool.submit(*t) for t in tasks]):
            pid, status = future.result()
            done[status] += 1
            if status != 'stopped':
                log(f'{label} {pid} {status} ({sum(done.values())}/{len(tasks)})')
    return dict(done)


def generate(mat, run, scratch, jobs):
    outdir = run / 'proposals'
    outdir.mkdir(exist_ok=True)
    tasks = [(generate_one, mat, outdir, j['id'], j['item'], j['arm'], scratch) for j in jobs]
    return run_pool('gen', mat.plan['proposer']['workers'], tasks)


def rate(mat, run, scratch, jobs, rater='claude'):
    outdir = run / ('ratings' if rater == 'claude' else f'ratings_{rater}')
    outdir.mkdir(exist_ok=True)
    order = sorted(jobs, key=lambda j: j['id'])
    random.Random(mat.plan['seed'] + 1).shuffle(order)
    tasks = [(rate_one, mat, outdir, j['id'], mat.key[j['item']], mat.packets[j['item']],
              run / 'proposals' / f"{j['id']}.txt", scratch, rater)
             for j in order if (run / 'proposals' / f"{j['id']}.txt").exists()]
    workers = (mat.plan['rater'] if rater == 'claude' else mat.second_rater)['workers']
    return run_pool(f'rate-{rater}', workers, tasks)


def sample(mat, run, jobs):
    rng = random.Random(mat.plan['seed'] + 2)
    cells = {}
    for j in jobs:
        cells.setdefault((j['arm'], mat.key[j['item']]['type']), []).append(j)
    chosen = []
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda j: j['id'])
        chosen += rng.sample(members, round(len(members) * mat.plan['scientist_fraction']))
    rng.shuffle(chosen)
    out = run / 'scientist'
    out.mkdir(exist_ok=True)
    lines = ['# EP24 blind scientist sample', '',
             'Score R1 and R3 with the rubric in SEARCH_POLICY.yaml and guess each '
             "proposal's arm (old, new, directive or unsure) in scores.csv. Do not open "
             '../sealed/ before scoring.', '']
    rows = []
    for j in chosen:
        path = run / 'proposals' / f"{j['id']}.txt"
        if not path.exists():
            continue
        k = mat.key[j['item']]
        lines += [f"## {j['id']} — {j['item']}, type {k['type']}", '', '### Packet', '', '~~~~text',
                  mat.packets[j['item']], '~~~~', '', '### Key', '', k['entry'], '',
                  '### Proposal', '', '~~~~text', path.read_text().strip(), '~~~~', '']
        rows.append([j['id'], j['item'], '', '', '', ''])
    (out / 'sample.md').write_text('\n'.join(lines))
    sheet = out / 'scores.csv'
    if not sheet.exists():
        with open(sheet, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'item', 'R1', 'R3', 'arm_guess', 'notes'])
            w.writerows(rows)
    return len(rows)


def test(mat, run, scratch):
    t = json.loads((mat.root / 'test_packet.json').read_text())
    outdir = run / 'test'
    outdir.mkdir(exist_ok=True)
    t0 = time.time()
    gen = run_pool('test-gen', 2, [(generate_one, mat, outdir, f'test_{arm}', t['item'], arm,
                                    scratch, t['packet']) for arm in ('old', 'new')])
    t1 = time.time()
    rates = run_pool('test-rate', 2, [(rate_one, mat, outdir, f'rating_test_{arm}', t, t['packet'],
                                       outdir / f'test_{arm}.txt', scratch)
                                      for arm in ('old', 'new') if (outdir / f'test_{arm}.txt').exists()])
    summary = {'generation': gen, 'rating': rates, 'generation_wall_s': round(t1 - t0, 1),
               'rating_wall_s': round(time.time() - t1, 1)}
    (outdir / 'timing.json').write_text(json.dumps(summary, indent=1) + '\n')
    log(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['test', 'generate', 'rate', 'sample', 'all'])
    ap.add_argument('--materials', required=True, help='outputs/arm_comparison directory')
    ap.add_argument('--run-dir', required=True, help='outputs/runs/arm_comparison/<attempt>')
    ap.add_argument('--scratch', required=True, help='transient working directory')
    ap.add_argument('--rater', choices=['claude', 'astra'], default='claude',
                    help='rate mode: Claude Opus (primary plan) or gpt-6-astra (second rater)')
    ap.add_argument('--plan', default='plan.json', help='plan file inside the materials directory')
    args = ap.parse_args()
    mat = Materials(args.materials, args.plan)
    run, scratch = Path(args.run_dir), Path(args.scratch)
    run.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if args.mode == 'test':
        return test(mat, run, scratch)
    jobs = schedule(mat)
    write_mapping(run, jobs)
    summary = {'planned': len(jobs)}
    if args.mode in ('generate', 'all'):
        summary['generation'] = generate(mat, run, scratch, jobs)
    if args.mode in ('rate', 'all'):
        summary[f'rating_{args.rater}'] = rate(mat, run, scratch, jobs, args.rater)
    if args.mode in ('sample', 'all'):
        summary['scientist_sample'] = sample(mat, run, jobs)
    (run / f'summary_{args.mode}_{time.strftime("%Y%m%dT%H%M%S")}.json').write_text(
        json.dumps(summary, indent=1) + '\n')
    log(json.dumps(summary))


if __name__ == '__main__':
    main()
