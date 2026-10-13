#!/usr/bin/env python3
"""Describe active-sform coordinate differences from completed EP04 headers only."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import socket
import time

import numpy as np


def active_sform(grid):
    if grid['sform_code'] <= 0:
        raise ValueError('Selected diagnostic requires a coded sform; no qform fallback')
    affine = np.eye(4)
    affine[:3] = np.asarray(grid['sform_rows'], dtype=float).reshape(3, 4)
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError('Nonfinite or singular active sform')
    return affine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use root-owned verified compute')
    if a.output_dir.exists():
        raise FileExistsError('Preserve previous attempts')
    source = json.loads(a.metadata_report.read_text())
    if not source['structural_pass_complete'] or len(source['betas']) != 150 or len(source['rois']) != 45:
        raise ValueError('Use the complete150-beta/45-ROI structural report')
    started = time.monotonic()
    betas = {}
    for beta in source['betas']:
        pid, grid = beta['participant'], beta['header']['spatial_grid_fields']
        affine = active_sform(grid)
        if pid in betas:
            old_grid, old_affine = betas[pid]
            if (grid['shape'] != old_grid['shape'] or grid['spatial_units_code'] != old_grid['spatial_units_code']
                    or not np.array_equal(affine, old_affine)):
                raise ValueError('Session-dependent beta active grid requires a separately specified diagnostic')
        else:
            betas[pid] = grid, affine
    result = []
    for roi in source['rois']:
        if roi['status'] != 'header_read_values_closed':
            raise ValueError('This observed complete panel requires every ROI header')
        grid = roi['header']['spatial_grid_fields']
        beta_grid, beta_affine = betas[roi['participant']]
        if grid['spatial_units_code'] != beta_grid['spatial_units_code']:
            raise ValueError('Mixed encoded spatial units; no implicit unit conversion')
        roi_affine = active_sform(grid)
        transform = np.linalg.solve(beta_affine, roi_affine)
        corners = np.asarray(list(itertools.product(*[(0., float(n-1)) for n in grid['shape']])))
        mapped = np.column_stack((corners, np.ones(8))) @ transform.T
        displacement = mapped[:, :3] - corners
        result.append({'participant': roi['participant'], 'ROI_path': roi['path'],
            'label': roi['label'], 'ROI_shape': grid['shape'], 'beta_shape': beta_grid['shape'],
            'sform_codes': {'ROI': grid['sform_code'], 'beta': beta_grid['sform_code']},
            'ROI_to_beta_voxel_transform': transform.tolist(),
            'coordinate_units': 'beta voxel index units; no physical-unit claim',
            'ROI_corner_indices': corners.tolist(), 'mapped_beta_corner_indices': mapped[:, :3].tolist(),
            'corner_displacements': displacement.tolist(),
            'maximum_absolute_coordinate_displacement_by_axis': np.max(np.abs(displacement), axis=0).tolist(),
            'maximum_absolute_coordinate_displacement': float(np.max(np.abs(displacement))),
            'maximum_Euclidean_displacement': float(np.max(np.linalg.norm(displacement, axis=1)))})
    report = {'analysis_id': 'source_readiness', 'job_id': os.environ['SLURM_JOB_ID'],
        'source_report': str(a.metadata_report), 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'completed': True, 'ROI_records': result, 'elapsed_seconds': time.monotonic()-started,
        'thresholds_applied': False, 'voxel_mapping_qualified': False, 'resampling_selected': False,
        'source_array_reads': 0, 'qform_used': False,
        'interpretation': 'Active-sform ROI-to-beta voxel displacement only. Each displacement coordinate is affine; its absolute maximum on the rectangular index domain occurs at a corner. No identity/resampling verdict is made.'}
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'roi_grid_offsets.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'completed': True, 'ROIs': len(result), 'thresholds_applied': False,
        'maximum_absolute_coordinate_displacement': max(r['maximum_absolute_coordinate_displacement'] for r in result),
        'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
