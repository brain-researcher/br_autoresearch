#!/usr/bin/env python3
"""Build the scientist-approved numerical common support and finish failed runs.

The original preparation remains unchanged. The support rule uses only original
BOLD finite-valuedness and positive run means, never fitted effects or designs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

import exploratory_fmri_runner as runner

RULE='all_timepoints_finite_and_positive_run_mean_for_every_432_run'
COMMON_FAILURE='ValueError: Nonfinite BOLD or nonpositive run mean inside frozen group mask; no group voxels filtered'


def selected_support(prepared, inventory, approved_support=None):
    """Resolve an explicit, fully built support; omission preserves the old mask."""
    original=Path(inventory['group_mask']).resolve()
    if approved_support is None:
        return original, int(inventory['group_mask_voxels']), {'selection':'original_frozen_support'}
    import numpy as np
    import nibabel as nib
    source=Path(approved_support).resolve()
    report=json.loads(source.read_text())
    if (report.get('status')!='complete_valid_common_support' or report.get('rule')!=RULE
            or report.get('prepared')!=str(prepared.resolve())
            or report.get('original_group_mask')!=str(original)
            or report.get('expected_runs')!=432 or report.get('verified_runs')!=432
            or report.get('unknown_runs')!=0
            or report.get('completed_run_certificates',-1)+report.get('fully_inspected_failed_runs',-1)!=432):
        raise ValueError('Approved support is incomplete or differs from the frozen preparation/rule')
    path=Path(report['mask']).resolve()
    old_image=nib.load(original); image=nib.load(path)
    old=np.asanyarray(old_image.dataobj); mask=np.asanyarray(image.dataobj)
    if (old.ndim!=3 or mask.shape!=old.shape or not np.allclose(image.affine,old_image.affine,rtol=0,atol=1e-5)
            or not np.isin(old,[0,1]).all() or not np.isin(mask,[0,1]).all()
            or np.any((mask==1)&(old!=1)) or not np.any(mask==1)):
        raise ValueError('Approved support must be a nonempty subset on the original mask grid')
    excluded=np.flatnonzero((old==1)&(mask!=1)).tolist()
    if (int((old==1).sum())!=inventory['group_mask_voxels']
            or int((mask==1).sum())!=report.get('mask_voxels')
            or excluded!=report.get('excluded_voxel_indices')
            or len(excluded)!=report.get('excluded_voxels')):
        raise ValueError('Approved mask and recorded original/count/exclusions disagree')
    return path, int(report['mask_voxels']), {
        'selection':'approved_valid_common_support','approved_support':str(source),
        'rule':RULE,'original_group_mask':str(original),'original_voxels':int((old==1).sum()),
        'retained_voxels':int((mask==1).sum()),'excluded_voxels':len(excluded),
        'amended_after_observed_input_failures':True,'original_support_analysis':'unavailable',
        'claim_scope':'Exploratory reanalysis on the explicitly amended numerical common support'}


def invalid_original_indices(values, mask, check=lambda:None, chunk_voxels=4096):
    """Inspect every original-common voxel and every acquired frame."""
    import numpy as np
    if values.ndim!=4 or values.shape[:3]!=mask.shape:
        raise ValueError('BOLD and original common-mask shapes differ')
    coordinates=np.where(mask)
    indices=np.flatnonzero(mask)
    invalid=[]
    counts={'nonfinite_series':0,'finite_nonpositive_mean_series':0,'all_zero_series':0}
    for start in range(0,len(indices),chunk_voxels):
        check(); stop=min(start+chunk_voxels,len(indices))
        y=np.asarray(values[coordinates[0][start:stop],coordinates[1][start:stop],coordinates[2][start:stop],:],dtype=float).T
        means=y.mean(0); finite=np.isfinite(y).all(0)&np.isfinite(means)
        valid=finite&(means>0)
        invalid.extend(indices[start:stop][~valid].tolist())
        counts['nonfinite_series']+=int((~finite).sum())
        counts['finite_nonpositive_mean_series']+=int((finite&(means<=0)).sum())
        counts['all_zero_series']+=int(np.all(y==0,axis=0).sum())
    return invalid,counts


def run_statuses(prepared, inventory, roots, check):
    """All 432 runs need a unique original completion or diagnosed failure."""
    from summarize_exploratory_fmri import validate_inventory, collect_completions
    _,entries,_=validate_inventory(inventory)
    completed,missing=collect_completions(prepared,roots,entries,check)
    original=str(Path(inventory['group_mask']).resolve())
    for key,(_,completion) in completed.items():
        if (str(Path(completion.get('group_mask','')).resolve())!=original
                or completion.get('group_mask_voxels')!=inventory['group_mask_voxels']
                or completion.get('approved_support') is not None):
            raise ValueError('Only an original-support successful fit certifies the entire original common mask')
    failures={}
    for missing_run in missing:
        check(); key=(missing_run['participant'],missing_run['run'])
        name=f'{key[0]}_run-{key[1]}.partial'
        paths=[root/name/'failure.json' for root in roots if (root/name/'failure.json').is_file()]
        if not paths:
            raise ValueError(f'Run still unattempted or running: {key[0]} run {key[1]}')
        for path in paths:
            failure=json.loads(path.read_text())
            if (failure.get('status')!='technical_failure' or failure.get('error')!=COMMON_FAILURE
                    or failure.get('participant')!=key[0] or failure.get('run')!=key[1]):
                raise ValueError(f'Unresolved failure other than the diagnosed common-mask guard: {path}')
        failures[key]=paths
    if len(completed)+len(failures)!=432:
        raise ValueError('Incomplete run-validity accounting')
    return entries,completed,failures


def build(prepared, roots, output, check, args):
    import numpy as np
    import nibabel as nib
    inventory=json.loads((prepared/'prepared.json').read_text())
    entries,completed,failures=run_statuses(prepared,inventory,roots,check)
    if output.exists():
        raise FileExistsError('Use a fresh support output directory')
    output.mkdir(parents=True)
    old_path=Path(inventory['group_mask']).resolve()
    image=nib.load(old_path); old=np.asanyarray(image.dataobj)
    if old.ndim!=3 or not np.isin(old,[0,1]).all() or int((old==1).sum())!=inventory['group_mask_voxels']:
        raise ValueError('Original frozen common mask changed')
    mask=old==1
    report={'analysis_id':'exploratory_fmri_valid_common_support','status':'building',
            'prepared':str(prepared.resolve()),'rule':RULE,'expected_runs':432,'verified_runs':len(completed),
            'unknown_runs':len(failures),'original_group_mask':str(old_path),
            'original_mask_voxels':int(mask.sum()),'completed_run_certificates':len(completed),
            'fully_inspected_failed_runs':0,'started_utc':datetime.now(timezone.utc).isoformat(),
            'run_roots':[str(root.resolve()) for root in roots],
            'failed_runs':[{'participant':key[0],'run':key[1]} for key in sorted(failures)],
            'run_checks':[],'excluded_voxel_indices':[],
            'mask_selection_uses_effects_or_response_associations':False,
            'support_amendment_after_observed_input_validity_failures':True}
    for key,(artifact,_) in sorted(completed.items()):
        report['run_checks'].append({'participant':key[0],'run':key[1],
            'status':'certified_by_original_support_completion','completion':str(artifact.parent/'completion.json')})
    excluded=set()
    for key,paths in sorted(failures.items()):
        check(); entry=entries[key]
        local=Path(os.environ['L_SCRATCH_JOB'])/f'ep01-valid-support-{os.getpid()}-{key[0]}-{key[1]}'
        try:
            staged=runner.stage_bold(entry['paths']['bold'],local,check,args.max_local_gib)
            bold,header=runner.image_header(staged,entry['paths']['bold_json'])
            if header!=entry['header'] or not np.allclose(bold.affine,image.affine,rtol=0,atol=1e-5):
                raise ValueError('Failed-run BOLD header/grid differs from original preparation')
            invalid,counts=invalid_original_indices(np.asanyarray(bold.dataobj),mask,check,args.chunk_voxels)
            if not invalid:
                raise ValueError('Prior common-mask failure did not reproduce on original input')
            excluded.update(invalid)
            report['run_checks'].append({'participant':key[0],'run':key[1],'status':'fully_scanned_original_common_support',
                'source_bold':entry['paths']['bold'],'prior_failures':[str(p) for p in paths],
                'invalid_voxel_indices':invalid,'invalid_voxels':len(invalid),**counts})
            report['fully_inspected_failed_runs']+=1
            report['verified_runs']+=1; report['unknown_runs']-=1
            report['excluded_voxel_indices']=sorted(excluded)
            runner.atomic_json(output/'support.progress.json',report)
            print(json.dumps({'stage':'support_scan','participant':key[0],'run':key[1],'invalid_voxels':len(invalid)}),flush=True)
        finally:
            if local.exists(): shutil.rmtree(local)
    valid=mask.copy(); valid.ravel()[sorted(excluded)]=False
    if not valid.any() or report['verified_runs']!=432 or report['unknown_runs']!=0:
        raise ValueError('No complete nonempty valid common support')
    path=output/'valid_common_mask.nii.gz'
    header=image.header.copy(); header.set_data_dtype('uint8')
    nib.save(nib.Nifti1Image(valid.astype('uint8'),image.affine,header),path)
    report.update(status='complete_valid_common_support',mask=str(path.resolve()),mask_voxels=int(valid.sum()),
                  excluded_voxels=len(excluded),excluded_voxel_indices=sorted(excluded),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    runner.atomic_json(output/'support.json',report)
    selected_support(prepared,inventory,output/'support.json')
    return report


def repair(prepared, approved_support, output, check, args):
    from summarize_exploratory_fmri import validate_inventory
    inventory=json.loads((prepared/'prepared.json').read_text())
    _,entries,_=validate_inventory(inventory)
    mask,_,metadata=selected_support(prepared,inventory,approved_support)
    support=json.loads(Path(approved_support).read_text())
    keys=[(row['participant'],row['run']) for row in support['failed_runs']]
    if len(set(keys))!=len(keys) or any(key not in entries for key in keys):
        raise ValueError('Approved support has duplicate or unknown failed-run keys')
    inspected={(row['participant'],row['run']) for row in support['run_checks']
               if row['status']=='fully_scanned_original_common_support'}
    if set(keys)!=inspected:
        raise ValueError('Repair menu differs from support-builder inspected failures')
    if output.exists(): raise FileExistsError('Use a fresh repair output root')
    output.mkdir(parents=True)
    report={'analysis_id':'exploratory_fmri_valid_support_repair','status':'running',
            'prepared':str(prepared.resolve()),'approved_support':str(Path(approved_support).resolve()),
            'selected_runs':support['failed_runs'],'completed_runs':[], 'spatial_support':metadata}
    runner.atomic_json(output/'repair.json',report)
    try:
        for key in keys:
            check()
            result=runner.fit_run(entries[key],prepared,output,check,args.chunk_voxels,args.max_local_gib,
                                  mask,approved_support=Path(approved_support))
            report['completed_runs'].append(result)
            runner.atomic_json(output/'repair.json',report)
            print(json.dumps(result),flush=True)
        report['status']='complete'
    except Exception as exc:
        report.update(status='technical_failure',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        runner.atomic_json(output/'repair.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','repair'])
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,action='append',default=[])
    parser.add_argument('--approved-support',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--max-rss-gib',type=float,default=12.)
    parser.add_argument('--max-local-gib',type=float,default=12.)
    parser.add_argument('--chunk-voxels',type=int,default=4096)
    args=parser.parse_args(); runner.require_compute()
    if min(args.max_seconds,args.max_rss_gib,args.max_local_gib,args.chunk_voxels)<=0:
        raise ValueError('Positive bounded runtime, memory and chunk limits required')
    began=time.monotonic()
    def check():
        if time.monotonic()-began>=args.max_seconds: raise TimeoutError('Support-stage runtime limit reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>args.max_rss_gib:
            raise MemoryError('Support-stage RSS limit reached')
    def interrupted(signum,frame): raise TimeoutError(f'Support-stage signal {signum}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(args.max_seconds)
    try:
        if args.command=='build':
            roots=[root.resolve() for root in args.run_root]
            if not roots or len(set(roots))!=len(roots): raise ValueError('Use unique explicit original run roots')
            result=build(args.prepared_dir,roots,args.output_dir,check,args)
        else:
            if args.approved_support is None: raise ValueError('Repair requires explicit --approved-support')
            result=repair(args.prepared_dir,args.approved_support,args.output_dir,check,args)
        check()
        print(json.dumps({'status':result['status'],'elapsed_seconds':time.monotonic()-began}),flush=True)
    finally: signal.alarm(0)

if __name__=='__main__': main()
