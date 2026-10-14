#!/usr/bin/env python3
"""Execute explicitly selected pending runs without changing EP01 scientific code.

Completed results are reused in place. A newly encountered frozen common-mask
validity failure remains unavailable and does not prevent another selected run
from being attempted. All other failures propagate. No failed run is retried.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import time

import exploratory_fmri_runner as runner

COMMON_MASK_FAILURE = ('Nonfinite BOLD or nonpositive run mean inside frozen group mask; '
                       'no group voxels filtered')


def key_of(record):
    person, run = record.get('participant'), record.get('run')
    if not isinstance(person,str) or type(run) is not int:
        raise ValueError('Selection requires a participant string and integer run')
    return person, run


def key_name(key):
    return f'{key[0]}_run-{key[1]}'


def selection_plan(inventory, selection, prepared, previous_roots):
    """Validate the exact run menu and locate only its already complete results."""
    if inventory.get('status')!='complete' or len(inventory.get('runs',[]))!=432:
        raise ValueError('Require the complete frozen 432-run preparation')
    entries={}
    for entry in inventory['runs']:
        key=key_of(entry)
        if key in entries:
            raise ValueError('Duplicate run in prepared inventory')
        entries[key]=entry
    if not isinstance(selection,list) or not selection:
        raise ValueError('Use a nonempty JSON list of explicit participant/run objects')
    keys=[]
    for record in selection:
        if not isinstance(record,dict) or set(record)!={'participant','run'}:
            raise ValueError('Each selection object must contain only participant and run')
        key=key_of(record)
        if key in keys:
            raise ValueError('Duplicate selected participant/run')
        if key not in entries:
            raise ValueError('Selected participant/run is outside the prepared panel')
        keys.append(key)
    roots=[Path(path).resolve() for path in previous_roots]
    if len(set(roots))!=len(roots):
        raise ValueError('Duplicate previous run root')
    if any(not root.is_dir() for root in roots):
        raise FileNotFoundError('A declared previous run root is absent')
    plan=[]
    for key in keys:
        matches=[]
        for root in roots:
            final=root/key_name(key)
            if final.exists():
                completion=json.loads((final/'completion.json').read_text())
                if (completion.get('status')!='complete' or completion.get('prepared')!=str(prepared.resolve())
                        or key_of(completion)!=key):
                    raise ValueError('Previous completion does not match this frozen prepared run')
                for artifact in completion['artifacts']:
                    if not (final/artifact).is_file():
                        raise FileNotFoundError('An artifact from a completed run is missing')
                matches.append(root)
            if (root/(key_name(key)+'.partial')).exists():
                raise ValueError('Selected run has a prior partial attempt; this continuation cannot retry it')
        if len(matches)>1:
            raise ValueError('Duplicate completed selected run across previous roots')
        plan.append({'entry':entries[key], 'previous_root':matches[0] if matches else None})
    return plan


def update_counts(report):
    records=report['runs']
    report['selected']=len(records)
    report['completed']=sum(r['status'] in ('complete','reused_complete') for r in records)
    report['unavailable']=sum(r['status']=='unavailable_common_mask' for r in records)
    report['pending']=sum(r['status']=='pending' for r in records)
    report['technical_failures']=sum(r['status']=='unexpected_failure' for r in records)


def execute_selection(prepared, selection_path, previous_roots, output, check, args):
    inventory=json.loads((prepared/'prepared.json').read_text())
    selection=json.loads(selection_path.read_text())
    plan=selection_plan(inventory,selection,prepared,previous_roots)
    if output.exists():
        raise FileExistsError('Continuation requires a fresh output root')
    output.mkdir(parents=True)
    report={'analysis_id':'exploratory_fmri_specification_sensitivity',
            'prepared':str(prepared.resolve()),'selection':str(selection_path.resolve()),
            'previous_run_roots':[str(Path(p).resolve()) for p in previous_roots],
            'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
            'scientific_runner':'exploratory_fmri_runner.fit_run',
            'unchanged_group_mask':inventory['group_mask'],
            'automatic_retry':False,'full_panel_required_runs':432,
            'full_panel_availability':'Not established by this selected-run continuation',
            'runs':[{'participant':p['entry']['participant'],'run':p['entry']['run'],'status':'pending'} for p in plan]}
    def checkpoint():
        update_counts(report)
        runner.atomic_json(output/'continuation.json',report)
    checkpoint()
    try:
        for item,record in zip(plan,report['runs']):
            check()
            entry=item['entry']
            previous=item['previous_root']
            record['status']='running'
            checkpoint()
            if previous is not None:
                # The plan already checked exact prepared/run identity and artifacts.
                record.update(status='reused_complete',run_directory=str(previous/key_name(key_of(entry))))
            else:
                try:
                    result=runner.fit_run(entry,prepared,output,check,args.chunk_voxels,
                                          args.max_local_gib,inventory['group_mask'])
                except ValueError as exc:
                    if str(exc)!=COMMON_MASK_FAILURE:
                        record.update(status='unexpected_failure',error=f'{type(exc).__name__}: {exc}')
                        raise
                    record.update(status='unavailable_common_mask',error=str(exc),
                                  failure=str(output/(key_name(key_of(entry))+'.partial')/'failure.json'))
                except Exception as exc:
                    record.update(status='unexpected_failure',error=f'{type(exc).__name__}: {exc}')
                    raise
                else:
                    record.update(status='complete',run_directory=str(output/key_name(key_of(entry))),
                                  elapsed_seconds=result['elapsed_seconds'])
            checkpoint()
            print(json.dumps(record),flush=True)
        check()
        update_counts(report)
        report['status']='selected_runs_complete' if report['unavailable']==0 else 'selected_runs_attempted_with_unavailable'
        return report, 0 if report['unavailable']==0 else 2
    except Exception as exc:
        report.update(status='technical_failure',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        checkpoint()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',required=True,type=Path)
    parser.add_argument('--selection-json',required=True,type=Path)
    parser.add_argument('--previous-run-root',action='append',default=[],type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--max-seconds',required=True,type=int)
    parser.add_argument('--max-rss-gib',type=float,default=12.)
    parser.add_argument('--max-local-gib',type=float,default=12.)
    parser.add_argument('--chunk-voxels',type=int,default=4096)
    args=parser.parse_args()
    runner.require_compute()
    if min(args.max_seconds,args.max_rss_gib,args.max_local_gib,args.chunk_voxels)<=0:
        raise ValueError('All process limits must be positive')
    began=time.monotonic()
    def check():
        if time.monotonic()-began>=args.max_seconds:
            raise TimeoutError('Whole-process continuation limit reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>args.max_rss_gib:
            raise MemoryError('Continuation RSS limit reached')
    def interrupted(signum,frame):
        raise TimeoutError(f'Continuation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(args.max_seconds)
    try:
        report,status=execute_selection(args.prepared_dir,args.selection_json,args.previous_run_root,
                                        args.output_dir,check,args)
        print(json.dumps({key:report[key] for key in ('status','selected','completed','unavailable','pending','technical_failures')}),flush=True)
        return status
    finally:
        signal.alarm(0)

if __name__=='__main__':
    raise SystemExit(main())
