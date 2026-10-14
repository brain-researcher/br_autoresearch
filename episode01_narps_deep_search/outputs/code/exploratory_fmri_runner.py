#!/usr/bin/env python3
"""Bounded exploratory NARPS first levels with one S00-derived AR1 per voxel.

Run only inside the coordinator's Slurm allocation. Scientific inputs are read
only; prepared inputs and each run completion are published atomically.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import time

EPISODE = Path(__file__).resolve().parents[2]
ARMS = ('S00', 'S10', 'S01', 'S11', 'R11')
FOCAL = ('gain', 'loss')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use a verified Slurm compute shell')
    if any(os.environ.get(name) != '1' for name in THREADS):
        raise RuntimeError('All numerical thread limits must equal one')


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_paths(participant, run):
    root = Path(os.environ['OAK']) / 'data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734'
    base = root / participant / 'func'
    name = f'{participant}_task-MGT_run-{run}'
    space = name + '_space-MNI152NLin2009cAsym_res-2'
    return {'bold': str(base / (space + '_desc-preproc_bold.nii.gz')),
            'bold_json': str(base / (space + '_desc-preproc_bold.json')),
            'mask': str(base / (space + '_desc-brain_mask.nii.gz')),
            'confounds': str(base / (name + '_desc-confounds_timeseries.tsv'))}


def confounds_for_run(path, n_scans):
    import numpy as np
    import pandas as pd
    data = pd.read_csv(path, sep='\t')
    if len(data) != n_scans:
        raise ValueError('Confound/BOLD timepoint mismatch')
    motion = [f'{axis}{suffix}' for axis in ('trans_x','trans_y','trans_z','rot_x','rot_y','rot_z')
              for suffix in ('','_derivative1','_power2','_derivative1_power2')]
    names = motion + ['white_matter', 'csf']
    if any(name not in data for name in names + ['framewise_displacement']):
        raise ValueError('Required 24-motion/WM/CSF/FD confound is missing')
    result = data[names].astype(float).copy()
    # Only undefined first-frame derivatives and FD have their documented zero.
    for name in motion:
        if '_derivative1' in name and np.isnan(result.loc[0, name]):
            result.loc[0, name] = 0.0
    fd = data['framewise_displacement'].to_numpy(dtype=float).copy()
    if np.isnan(fd[0]):
        fd[0] = 0.0
    if not np.isfinite(result.to_numpy()).all() or not np.isfinite(fd).all():
        raise ValueError('Unexpected missing/nonfinite confound value')
    spike_rows = set(np.flatnonzero(fd > .5).tolist())
    ns_names = sorted(name for name in data if name.startswith('non_steady_state_outlier'))
    for name in ns_names:
        values = data[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, [0,1]).all():
            raise ValueError('Invalid nonsteady-state spike indicator')
        spike_rows.update(np.flatnonzero(values).tolist())
    for row in sorted(spike_rows):
        values = np.zeros(n_scans)
        values[row] = 1.0
        result[f'censor_spike_{row:04d}'] = values
    return result, {'motion_columns': motion, 'tissue_columns': ['white_matter','csf'],
                    'FD_threshold': .5, 'FD_spike_count': int((fd > .5).sum()),
                    'nonsteady_source_columns': ns_names, 'unique_spike_rows': sorted(spike_rows),
                    'confound_columns': list(result.columns)}


def image_header(path, metadata_path=None):
    import nibabel as nib
    import numpy as np
    image = nib.load(path)
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError('Require a four-dimensional BOLD run')
    spatial, temporal = image.header.get_xyzt_units()
    if spatial != 'mm' or temporal not in ('sec','msec','usec'):
        raise ValueError(f'Unsupported NIfTI units: {spatial}, {temporal}')
    tr = float(image.header.get_zooms()[3]) * {'sec':1.,'msec':.001,'usec':.000001}[temporal]
    if not np.isfinite(tr) or tr <= 0 or not np.isfinite(image.affine).all():
        raise ValueError('Invalid NIfTI TR/affine')
    path = Path(path)
    if metadata_path is None:
        metadata_path = path.with_name(path.name.removesuffix('.nii.gz').removesuffix('.nii')+'.json')
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get('SliceTimingCorrected') is not False:
        raise ValueError('Expected explicitly uncorrected slice timing for volume-onset convention')
    json_tr = metadata.get('RepetitionTime')
    if not isinstance(json_tr,(int,float)) or not np.isclose(json_tr,tr,rtol=0,atol=1e-6):
        raise ValueError('Derivative JSON and NIfTI TR disagree')
    return image, {'shape': list(image.shape), 'affine': image.affine.tolist(),
                   'tr_seconds': tr,
                   'slice_timing_corrected':False, 'frame_origin':'volume_onset_no_exact_slice_alignment_claim',
                   'zooms': list(map(float,image.header.get_zooms())),
                   'datatype': str(image.get_data_dtype())}


def prepare(output, check, slice_time_ref):
    import numpy as np
    import nibabel as nib
    from run_observed_behavior import read_source
    from exploratory_fmri_design import behavioral_covariates, build_design
    if output.exists():
        raise FileExistsError('Prepare requires a fresh output directory')
    output.mkdir(parents=True)
    people = read_source(check)
    people, covariate_metadata = behavioral_covariates(people, check)
    atomic_json(output / 'behavioral_covariates.json', {'people':people, 'metadata':covariate_metadata})
    inventory = {'analysis_id':'exploratory_observed_fmri', 'status':'preparing',
                 'participants': list(people), 'runs':[], 'pilot_participants':[],
                 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'behavioral_covariates': 'behavioral_covariates.json',
                 'AR_strategy':'S00 voxelwise AR1 coefficients, reused for all arms',
                 'mask_strategy':'per-participant intersection of four input masks',
                 'signal_scale':'100*(voxel timecourse / its run mean - 1)',
                 'smoothing_fwhm_mm':0., 'arms':list(ARMS)}
    inventory['slice_time_ref_TR_fraction'] = slice_time_ref
    global_mask, global_affine, global_header = None, None, None
    for version in ('EI','ER'):
        inventory['pilot_participants'].append(next(p for p in people if people[p]['version']==version))
    for person, panel in people.items():
        check()
        person_dir = output / person
        person_dir.mkdir()
        mask, reference, reference_header = None, None, None
        person_runs = []
        for run in range(1,5):
            check()
            paths = source_paths(person, run)
            for path in paths.values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            image, header = image_header(paths['bold'])
            mask_image = nib.load(paths['mask'])
            if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-5):
                raise ValueError('Mask and BOLD grids differ')
            values = np.asanyarray(mask_image.dataobj)
            if not np.isfinite(values).all() or not np.isin(values,[0,1]).all():
                raise ValueError('Mask must contain finite zero/one values')
            current = values > 0
            if mask is None:
                mask = current.copy()
                reference = image.affine.copy()
                reference_header = image.header.copy()
            elif mask.shape != current.shape or not np.allclose(reference,image.affine,rtol=0,atol=1e-5):
                raise ValueError('Participant run grids differ')
            else:
                mask &= current
            nuisance, nuisance_meta = confounds_for_run(paths['confounds'],image.shape[3])
            rows = [r for r in panel['rows'] if int(r['run'])==run]
            frame_times = (np.arange(image.shape[3], dtype=float)+slice_time_ref)*header['tr_seconds']
            designs, diagnostics = {}, {}
            for arm in ARMS:
                check()
                matrix, diagnostic = build_design(rows, frame_times, nuisance, arm, check)
                if not np.isfinite(matrix.to_numpy()).all():
                    raise ValueError('Nonfinite design matrix')
                design_file = person_dir / f'run-{run}_{arm}_design.tsv'
                matrix.to_csv(design_file,sep='\t',index=False)
                designs[arm] = str(design_file.resolve())
                diagnostics[arm] = diagnostic
            # Preserve all designs, irrespective of quality thresholds.
            for arm, reference_arm in (('S11','S00'),('R11','S11')):
                ratios = {}
                for focal in FOCAL:
                    current = diagnostics[arm]['focal'][focal]['efficiency']
                    base = diagnostics[reference_arm]['focal'][focal]['efficiency']
                    ratios[focal] = float(current/base) if current is not None and base is not None and base>0 else None
                diagnostics[arm]['efficiency_relative_to_'+reference_arm] = ratios
                diagnostics[arm]['relative_efficiency_pass'] = all(value is not None and value>=.5 for value in ratios.values())
            diagnostic_path = person_dir / f'run-{run}_design_diagnostics.json'
            atomic_json(diagnostic_path, {'arms': diagnostics, 'nuisance':nuisance_meta})
            entry = {'participant':person, 'task_version':panel['version'], 'run':run,
                     'paths':paths, 'header':header, 'designs':designs,
                     'design_diagnostics':str(diagnostic_path.resolve())}
            person_runs.append(entry)
        if mask is None or not mask.any():
            raise ValueError(f'{person}: empty four-run intersection mask')
        mask_path = person_dir / 'four_run_intersection_mask.nii.gz'
        reference_header.set_data_dtype('uint8')
        nib.save(nib.Nifti1Image(mask.astype('uint8'), reference, reference_header),mask_path)
        if global_mask is None:
            global_mask, global_affine, global_header = mask.copy(), reference.copy(), reference_header.copy()
        elif global_mask.shape != mask.shape or not np.allclose(global_affine,reference,rtol=0,atol=1e-5):
            raise ValueError('Participant grids differ; cannot construct fixed group intersection')
        else:
            global_mask &= mask
        for entry in person_runs:
            entry.update(mask=str(mask_path.resolve()), mask_voxels=int(mask.sum()))
        inventory['runs'].extend(person_runs)
        print(json.dumps({'stage':'prepared_participant','participant':person,'mask_voxels':int(mask.sum())}),flush=True)
    if len(inventory['runs'])!=432:
        raise ValueError('Preparation did not retain all 432 runs')
    if global_mask is None or not global_mask.any():
        raise ValueError('All-432-run group intersection is empty')
    global_path = output/'all_432_run_intersection_mask.nii.gz'
    nib.save(nib.Nifti1Image(global_mask.astype('uint8'),global_affine,global_header),global_path)
    inventory.update(group_mask=str(global_path.resolve()), group_mask_voxels=int(global_mask.sum()),
                     group_summary_support='Fixed all-432-run intersection only; no varying participant counts or zero filling')
    inventory.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / 'prepared.json', inventory)
    return {'prepared_runs':432, 'pilot_participants':inventory['pilot_participants']}


def independent_design(matrix):
    """Full-rank basis plus estimable original focal contrasts.

    A rank-deficient original coefficient is never reported as its arbitrary
    minimum-norm value. A basis also gives Nilearn the correct residual DOF.
    """
    import numpy as np
    values = matrix.to_numpy(dtype=float)
    u, singular, vt = np.linalg.svd(values, full_matrices=False)
    tolerance = singular[0] * max(values.shape) * np.finfo(float).eps
    rank = int((singular > tolerance).sum())
    if rank == 0 or rank >= len(values):
        raise ValueError('No residual degrees of freedom')
    basis = u[:,:rank]
    transform = vt[:rank,:].T / singular[:rank]
    projector = vt[:rank,:].T @ vt[:rank,:]
    contrasts, estimable = {}, {}
    for focal in FOCAL:
        index = list(matrix.columns).index(focal)
        target = np.eye(values.shape[1])[index]
        valid = bool(np.linalg.norm(target-projector[index]) <= 1e-8)
        estimable[focal] = valid
        contrasts[focal] = transform[index] if valid else None
    return basis, contrasts, {'rank':rank,'columns':values.shape[1],
                            'residual_degrees_of_freedom':len(values)-rank,
                            'focal_estimable':estimable,'rank_tolerance':float(tolerance)}


def fit_shared_ar_chunk(Y, designs, check=lambda:None, model_cache=None):
    """AR1 bin labels are estimated once from S00 and shared across all arms."""
    import numpy as np
    from nilearn.glm.first_level import run_glm
    from nilearn.glm.regression import ARModel
    check()
    labels, baseline = run_glm(Y, designs['S00'][0], noise_model='ar1',bins=100,n_jobs=1)
    if model_cache is None:
        model_cache = {}
    output = {}
    for arm in ARMS:
        check()
        basis, contrasts, _ = designs[arm]
        effect = np.full((2,Y.shape[1]),np.nan)
        variance = np.full_like(effect,np.nan)
        for label, reference in baseline.items():
            selected = labels == label
            if arm=='S00':
                fit = reference
            else:
                cache_key = (arm,str(label))
                if cache_key not in model_cache:
                    model_cache[cache_key] = ARModel(basis,reference.model.rho.copy())
                fit = model_cache[cache_key].fit(Y[:,selected])
            for index,focal in enumerate(FOCAL):
                contrast = contrasts[focal]
                if contrast is not None:
                    effect[index,selected] = contrast @ fit.theta
                    variance[index,selected] = float(contrast @ fit.cov @ contrast) * fit.dispersion
                    if not np.isfinite(effect[index,selected]).all() or not np.isfinite(variance[index,selected]).all() or (variance[index,selected]<0).any():
                        raise FloatingPointError('Nonfinite effect or invalid effect variance')
            if arm!='S00':
                del fit
        output[arm] = effect,variance
    rho = np.array([float(label) for label in labels],dtype='float32')
    return output,rho


def stage_bold(source, local, check, max_local_gib):
    """Stream one gzip file to node-local uncompressed NIfTI for mmap access."""
    local.mkdir(parents=True,exist_ok=False)
    destination = local / 'bold.nii'
    limit = int(max_local_gib * 1024**3)
    if shutil.disk_usage(local).free < limit:
        raise OSError('Insufficient local scratch for declared per-run staging limit')
    used = 0
    with gzip.open(source,'rb') as reader, destination.open('xb') as writer:
        while True:
            check()
            block = reader.read(8*1024**2)
            if not block:
                break
            used += len(block)
            if used > limit:
                raise OSError('Uncompressed BOLD exceeds local staging limit')
            writer.write(block)
    return destination


def fit_bold_chunk(Y, in_group_mask, designs, check=lambda:None, model_cache=None):
    """Retain unavailable outside-group voxels; never alter fixed group support."""
    import numpy as np
    means = Y.mean(axis=0)
    valid = np.isfinite(Y).all(axis=0) & np.isfinite(means) & (means > 0)
    if np.any(~valid & in_group_mask):
        raise ValueError('Nonfinite BOLD or nonpositive run mean inside frozen group mask; no group voxels filtered')
    effects = np.full((len(ARMS),2,Y.shape[1]),np.nan)
    variances = np.full_like(effects,np.nan)
    rho = np.full(Y.shape[1],np.nan,dtype='float32')
    if valid.any():
        scaled = 100*(Y[:,valid]/means[valid]-1.)
        fitted, ar = fit_shared_ar_chunk(scaled,designs,check,model_cache)
        for index,arm in enumerate(ARMS):
            effects[index][:,valid],variances[index][:,valid] = fitted[arm]
        rho[valid] = ar
    return effects,variances,rho,valid


def fit_run(entry, prepared, output, check, chunk_voxels, max_local_gib, group_mask_path, approved_support=None):
    import numpy as np
    import pandas as pd
    import nibabel as nib
    if approved_support is not None:
        from exploratory_fmri_support import selected_support
        original_inventory=json.loads((prepared/'prepared.json').read_text())
        selected,_,_=selected_support(prepared,original_inventory,approved_support)
        if Path(group_mask_path).resolve()!=selected:
            raise ValueError('Fit mask differs from explicitly approved support')
    key = f"{entry['participant']}_run-{entry['run']}"
    final = output / key
    if final.exists():
        completion = json.loads((final/'completion.json').read_text())
        if (completion.get('status') != 'complete' or completion.get('prepared')!=str(prepared.resolve())
                or completion.get('participant')!=entry['participant'] or completion.get('run')!=entry['run']):
            raise ValueError('Existing run cannot be reused as this exact prepared run')
        if approved_support is not None and Path(completion['group_mask']).resolve() not in {
                Path(original_inventory['group_mask']).resolve(), Path(group_mask_path).resolve()}:
            raise ValueError('Previous run used a different spatial support')
        for name in completion['artifacts']:
            if not (final/name).is_file():
                raise FileNotFoundError('Completed run artifact is missing')
        return {'run':key,'status':'reused_complete','elapsed_seconds':0.}
    work = output / (key+'.partial')
    if work.exists():
        raise FileExistsError('Partial attempt preserved; select a fresh output root')
    work.mkdir()
    began = time.monotonic()
    local_root = Path(os.environ['L_SCRATCH_JOB'])
    local = local_root / f"ep01-fmri-{os.getpid()}-{key}"
    try:
        designs = {arm: independent_design(pd.read_csv(entry['designs'][arm],sep='\t')) for arm in ARMS}
        mask_image = nib.load(entry['mask'])
        mask = np.asanyarray(mask_image.dataobj)>0
        group_image = nib.load(group_mask_path)
        group_mask = np.asanyarray(group_image.dataobj)>0
        if (group_mask.shape != mask.shape or not np.allclose(group_image.affine,mask_image.affine,rtol=0,atol=1e-5)
                or np.any(group_mask & ~mask)):
            raise ValueError('Frozen group support is not contained in the participant mask')
        coords = np.where(mask)
        voxel_indices = np.flatnonzero(mask)
        common = group_mask[coords]
        n_voxels = len(coords[0])
        staged = stage_bold(entry['paths']['bold'],local,check,max_local_gib)
        image, header = image_header(staged,entry['paths']['bold_json'])
        if header != entry['header']:
            raise ValueError('BOLD header differs from prepared input')
        values = np.asanyarray(image.dataobj)
        check()
        effects = np.full((len(ARMS),2,n_voxels),np.nan,dtype='float32')
        variances = np.full_like(effects,np.nan)
        rho = np.full(n_voxels,np.nan,dtype='float32')
        invalid_outside_group = []
        model_cache = {}
        for start in range(0,n_voxels,chunk_voxels):
            check()
            stop = min(start+chunk_voxels,n_voxels)
            Y = np.asarray(values[coords[0][start:stop],coords[1][start:stop],coords[2][start:stop],:],dtype=float).T
            chunk_effects,chunk_variances,ar,valid = fit_bold_chunk(Y,common[start:stop],designs,check,model_cache)
            effects[:,:,start:stop],variances[:,:,start:stop] = chunk_effects,chunk_variances
            rho[start:stop] = ar
            invalid_outside_group.extend(voxel_indices[start:stop][~valid].tolist())
        check()
        artifact = work/'gain_loss_effects.npz'
        np.savez_compressed(artifact, effects=effects, variances=variances, ar1=rho,
                            voxel_indices=voxel_indices, shape=np.array(mask.shape),
                            affine=image.affine, arms=np.array(ARMS), focal=np.array(FOCAL))
        shutil.copyfile(entry['mask'],work/'run_mask.nii.gz')
        completion = {'status':'complete','participant':entry['participant'],'task_version':entry['task_version'],
                      'run':entry['run'],'prepared':str(prepared.resolve()),'source':entry['paths'],
                      'voxel_count':n_voxels,'designs':entry['designs'],'design_diagnostics':entry['design_diagnostics'],
                      'fit_diagnostics':{arm:designs[arm][2] for arm in ARMS},
                      'shared_AR1_counts':dict(Counter(map(str,rho[np.isfinite(rho)].tolist()))),
                      'group_mask':str(group_mask_path), 'group_mask_voxels':int(common.sum()),
                      'invalid_outside_group_mask':{'voxel_count':len(invalid_outside_group),
                          'voxel_indices':invalid_outside_group,
                          'handling':'Full participant mask retained; effects, variances and AR1 explicitly NaN only outside frozen group support',
                          'reason':'Nonfinite BOLD series or nonpositive run mean'},
                      'effect_units':'percent signal change per physical offer unit',
                      'variance_units':'squared percent signal change per squared physical offer unit',
                      'artifacts':['gain_loss_effects.npz','run_mask.nii.gz'],
                      'elapsed_seconds':time.monotonic()-began,
                      'completed_utc':datetime.now(timezone.utc).isoformat()}
        if approved_support is not None:
            completion['approved_support']=str(Path(approved_support).resolve())
            completion['original_group_mask']=original_inventory['group_mask']
        atomic_json(work/'completion.json',completion)
        check()
        work.rename(final)
        return {'run':key,'status':'complete','elapsed_seconds':completion['elapsed_seconds'],
                'voxel_count':n_voxels}
    except Exception as exc:
        atomic_json(work/'failure.json',{'status':'technical_failure','error':f'{type(exc).__name__}: {exc}',
                                       'elapsed_seconds':time.monotonic()-began,'participant':entry['participant'],'run':entry['run']})
        raise
    finally:
        if local.exists():
            shutil.rmtree(local)


def fit(prepared, output, participants, check, args):
    inventory = json.loads((prepared/'prepared.json').read_text())
    if inventory.get('status')!='complete' or len(inventory.get('runs',[]))!=432:
        raise ValueError('Require complete all-run preparation')
    if not participants or len(set(participants))!=len(participants) or not set(participants)<=set(inventory['participants']):
        raise ValueError('Explicit unique prepared participant IDs required')
    approved_support=getattr(args,'approved_support',None)
    group_mask=inventory['group_mask']
    if approved_support is not None:
        from exploratory_fmri_support import selected_support
        group_mask,_,_=selected_support(prepared,inventory,approved_support)
    output.mkdir(parents=True,exist_ok=True)
    result=[]
    for entry in inventory['runs']:
        if entry['participant'] not in participants:
            continue
        check()
        record=fit_run(entry,prepared,output,check,args.chunk_voxels,args.max_local_gib,group_mask,
                       approved_support=approved_support)
        result.append(record)
        print(json.dumps(record),flush=True)
    return {'selected_participants':participants,'runs':result,'status':'complete'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','fit'])
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--prepared-dir',type=Path)
    parser.add_argument('--participant',action='append',default=[])
    parser.add_argument('--max-seconds',required=True,type=int)
    parser.add_argument('--max-rss-gib',type=float,default=12.)
    parser.add_argument('--max-local-gib',type=float,default=12.)
    parser.add_argument('--chunk-voxels',type=int,default=4096)
    parser.add_argument('--slice-time-ref',type=float,default=0.)
    parser.add_argument('--approved-support',type=Path)
    args=parser.parse_args()
    require_compute()
    if args.max_seconds<=0 or args.chunk_voxels<=0 or args.max_rss_gib<=0 or args.max_local_gib<=0:
        raise ValueError('Limits must be positive')
    began=time.monotonic()
    def check():
        if time.monotonic()-began >= args.max_seconds:
            raise TimeoutError('Whole-process runtime limit reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > args.max_rss_gib:
            raise MemoryError('Declared process RSS limit reached')
    def interrupted(signum,frame):
        raise TimeoutError(f'Runtime interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(args.max_seconds)
    status=0
    try:
        if args.command=='prepare':
            if args.approved_support is not None:
                raise ValueError('Approved support is separate from immutable original preparation')
            if args.slice_time_ref != 0.:
                raise ValueError('Frozen uncorrected-slice volume-onset reference must be zero')
            result=prepare(args.output_dir,check,args.slice_time_ref)
        else:
            if args.prepared_dir is None:
                raise ValueError('--prepared-dir required for fit')
            result=fit(args.prepared_dir,args.output_dir,args.participant,check,args)
        check()
        print(json.dumps({'status':'complete','elapsed_seconds':time.monotonic()-began,**result}),flush=True)
    except Exception as exc:
        print(json.dumps({'status':'technical_failure','error':f'{type(exc).__name__}: {exc}',
                          'elapsed_seconds':time.monotonic()-began}),flush=True)
        raise
    finally:
        signal.alarm(0)
    return status

if __name__=='__main__':
    raise SystemExit(main())
