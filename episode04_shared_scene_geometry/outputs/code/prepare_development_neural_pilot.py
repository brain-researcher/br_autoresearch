#!/usr/bin/env python3
"""Fixed sub-01/ses-01 trusted selected-row beta/ROI handoff; no scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import struct
import time

import numpy as np

from inspect_array_metadata import EVC_COMPONENTS, PARTICIPANTS, SECTORS, parse_header, read_json


PARTICIPANT, SESSION = 'sub-01', 'ses-01'
EPISODE = 'episode04_shared_scene_geometry'
SPATIAL_SHAPE = [143, 143, 143]
SCRATCH_LIMIT = 16 * 1024**3
ROI_NAMES = (*SECTORS, 'retinotopic-EVC')
HEADER_FIELDS = ('shape', 'datatype_code', 'bits_per_voxel', 'voxel_offset',
                 'header_bytes_decompressed', 'scl_slope', 'scl_inter', 'spatial_grid_fields')


def read_exact(stream, count):
    """Return bytes only; callers alone decide whether a volume is numeric."""
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError('Truncated NIfTI stream')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def open_header(stream, expected):
    first = read_exact(stream, 4)
    sizes = [(order, struct.unpack(order + 'i', first)[0]) for order in ('<', '>')]
    supported = [(order, size) for order, size in sizes if size in (348, 540)]
    if len(supported) != 1:
        raise ValueError('Unsupported NIfTI header size/endian')
    order, size = supported[0]
    header = parse_header(first + read_exact(stream, size - 4))
    if any(header[key] != expected[key] for key in HEADER_FIELDS):
        raise ValueError('Current NIfTI header differs from the frozen structural record')
    # These exact exports were qualified as float32 with identity scaling.
    # Fail rather than silently choose a new dtype/scaling interpretation.
    if (header['datatype_code'] != 16 or header['bits_per_voxel'] != 32
            or header['scl_slope'] != 1.0 or header['scl_inter'] != 0.0):
        raise ValueError('This fixed handoff requires the qualified float32 identity scaling')
    offset = header['voxel_offset']
    if offset != int(offset) or not size <= offset <= 1024 * 1024:
        raise ValueError('Invalid or unsupported NIfTI voxel offset')
    read_exact(stream, int(offset) - size)  # Extension/padding remains opaque.
    return header, np.dtype(order + 'f4')


def mask_membership(values):
    if np.isinf(values).any():
        raise ValueError('Infinite ROI mask value is malformed')
    return np.isfinite(values) & (values != 0)


def read_mask(path, expected):
    with gzip.open(path, 'rb') as stream:
        header, dtype = open_header(stream, expected)
        if len(header['shape']) != 3:
            raise ValueError('Expected a3D fixed ROI mask')
        raw = read_exact(stream, 4 * math.prod(header['shape']))
        values = np.frombuffer(raw, dtype=dtype).reshape(header['shape'], order='F').ravel(order='C')
        mask = mask_membership(values)
        nan_count = int(np.isnan(values).sum())
        if stream.read(1):
            raise ValueError('Unexpected ROI payload after the specified volume')
    return mask, nan_count


def storage_indices(c_indices, spatial_shape):
    """Provider C-flat voxel identities -> NIfTI's X-fastest storage index."""
    coordinates = np.unravel_index(c_indices, spatial_shape, order='C')
    return np.ravel_multi_index(coordinates, spatial_shape, order='F')


def decode_selected(raw, dtype, selected_storage_indices):
    # Invoked only after source trial membership is authenticated.
    return np.frombuffer(raw, dtype=dtype)[selected_storage_indices]


class SequentialNpyRows:
    """Write ascending selected rows without retaining mapped output pages."""
    def __init__(self, stream, shape):
        self.stream, self.shape = stream, shape
        self.dtype, self.next_row = np.dtype('float32'), 0
        np.lib.format.write_array_header_2_0(stream, {
            'descr': '<f4', 'fortran_order': False, 'shape': shape})

    def __setitem__(self, row, values):
        if row != self.next_row or values.shape != (self.shape[1],):
            raise ValueError('NPY payload rows must be written once in frozen ascending order')
        self.stream.write(np.asarray(values, dtype='<f4').tobytes(order='C'))
        self.next_row += 1


def validate_session_rows(rows, trial_count):
    if not rows:
        raise ValueError('No selected development trials for the fixed session')
    indices = [row['session_beta_axis_index'] for row in rows]
    if (any(type(index) is not int or not 0 <= index < trial_count for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Session beta-axis positions are duplicate or out of range')
    # run-local beta_index is deliberately not the fourth-axis coordinate.
    return sorted(rows, key=lambda row: row['session_beta_axis_index'])


def stream_selected(stream, shape, dtype, rows, union_indices, destination,
                    progress, checkpoint=None, decoder=decode_selected):
    rows = validate_session_rows(rows, shape[3])
    if destination.shape != (len(rows), len(union_indices)) or destination.dtype != np.dtype('float32'):
        raise ValueError('Selected output shape/dtype disagree')
    by_axis = {row['session_beta_axis_index']: output for output, row in enumerate(rows)}
    positions = storage_indices(union_indices, shape[:3])
    bytes_per_volume = 4 * math.prod(shape[:3])
    for source_axis in range(shape[3]):
        raw = read_exact(stream, bytes_per_volume)
        progress['source_volumes_consumed'] += 1
        progress['uncompressed_beta_payload_bytes_consumed'] += len(raw)
        if source_axis in by_axis:
            values = decoder(raw, dtype, positions)
            if values.shape != (len(union_indices),) or values.dtype.kind != 'f' or values.dtype.itemsize != 4:
                raise ValueError('Selected numeric volume has unexpected dtype or shape')
            destination[by_axis[source_axis]] = values
            progress['selected_volumes_decoded'] += 1
            progress['selected_roi_nonfinite_values'] += int((~np.isfinite(values)).sum())
            progress['selected_roi_zero_values'] += int((values == 0).sum())
        else:
            # No np.frombuffer/array/summary of this protected response volume.
            progress['excluded_volumes_discarded_as_opaque_bytes'] += 1
        if checkpoint is not None and (source_axis + 1) % 64 == 0:
            checkpoint()
    if stream.read(1):
        raise ValueError('Unexpected beta payload after the specified fourth axis')
    if progress['selected_volumes_decoded'] != len(rows):
        raise ValueError('Incomplete selected development rows')


def frozen_inputs(metadata_dir, array_metadata, mapping_path,
                  participant=PARTICIPANT, session=SESSION):
    if participant not in PARTICIPANTS or session not in {f'ses-{n:02d}' for n in range(1, 31)}:
        raise ValueError('Only the frozen five participants and30 sessions are allowed')
    summary = read_json(metadata_dir / 'source_metadata_report.json')
    arrays = read_json(array_metadata)
    mapping = read_json(mapping_path)
    if (summary.get('metadata_fold_freeze_complete') is not True
            or summary.get('development_images') != 23560
            or summary.get('development_trials') != 94216
            or arrays.get('structural_pass_complete') is not True
            or mapping.get('mode') != 'provider_native_same_array_index'
            or mapping.get('array_shape') != SPATIAL_SHAPE
            or mapping.get('resampling') != 'none'):
        raise ValueError('Completed frozen metadata and adopted index mapping required')
    images = {}
    with (metadata_dir / 'development_images.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['image_id'] in images:
                raise ValueError('Duplicate development image identity')
            images[row['image_id']] = row
    if len(images) != 23560:
        raise ValueError('Incomplete frozen development image set')
    rows, total_rows = [], 0
    with (metadata_dir / 'development_trials.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            total_rows += 1
            if (row['participant'], row['session']) != (participant, session):
                continue
            image = images.get(row['image_id'])
            if (image is None or any(row[key] != image[key] for key in
                    ('participant', 'fold', 'component_id', 'source_stratum'))
                    or type(row['fold']) is not int or row['fold'] not in range(5)):
                raise ValueError('Selected trial does not match the frozen development image')
            rows.append(row)
    if total_rows != summary['development_trials']:
        raise ValueError('Incomplete frozen development trial export')
    trial_records = [r for r in summary['trial_files']
                     if (r['participant'], r['session']) == (participant, session)]
    betas = [r for r in arrays['betas']
             if (r['participant'], r['session']) == (participant, session)]
    if len(trial_records) != 1 or len(betas) != 1:
        raise ValueError('Expected one frozen beta/trial record for the selected session')
    if len(rows) != trial_records[0]['role_counts']['development_unique']:
        raise ValueError('Selected session development count differs from frozen role accounting')
    beta = betas[0]
    expected_beta = (f'derivatives/glmsingle-tedana/{participant}/{session}/func/'
                     f'{participant}_{session}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz')
    if (beta['path'] != expected_beta
            or beta['header']['shape'] != SPATIAL_SHAPE + [trial_records[0]['rows']]
            or beta['original_tsv_rows'] != trial_records[0]['rows']
            or beta['sidecar'].get('GLMsingleField') != 'betasmd'
            or beta['sidecar'].get('ModelStage') != 'TYPED'):
        raise ValueError('Fixed session beta identity/representation/axis changed')
    rows = validate_session_rows(rows, trial_records[0]['rows'])
    rois = []
    for family, labels in (('laion', SECTORS), ('retinotopy', EVC_COMPONENTS)):
        for label in labels:
            exact = (f'derivatives/rois/{participant}/{family}/'
                     f'{participant}_space-T1w_res-1pt8_label-{label}_mask.nii.gz')
            matches = [r for r in arrays['rois'] if r['path'] == exact]
            if (len(matches) != 1 or matches[0]['header']['shape'] != SPATIAL_SHAPE
                    or matches[0].get('selected_in_completed_acquisition') is not True):
                raise ValueError('A fixed ROI component lacks the qualified header')
            rois.append(matches[0])
    return summary, arrays, beta, rows, rois


def fresh_payload_directory(path):
    scratch = os.environ.get('SCRATCH')
    if not scratch:
        raise ValueError('SCRATCH must identify the authorized transient workspace')
    allowed = (Path(scratch) / 'br_autoresearch' / EPISODE).resolve()
    resolved = path.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed) or path.exists():
        raise ValueError('Use a fresh child path under the episode nonlegacy SCRATCH directory')
    path.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--array-metadata', type=Path, required=True)
    parser.add_argument('--roi-mapping', type=Path, required=True)
    parser.add_argument('--payload-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=540)
    parser.add_argument('--participant', choices=PARTICIPANTS, default=PARTICIPANT)
    parser.add_argument('--session', choices=[f'ses-{n:02d}' for n in range(1, 31)], default=SESSION)
    parser.add_argument('--scratch-limit-bytes', type=int, default=SCRATCH_LIMIT)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    if (args.output_dir.exists() or not 0 < args.max_seconds <= 540
            or not 0 < args.scratch_limit_bytes <= SCRATCH_LIMIT):
        raise ValueError('Fresh report path, deadline<=540s and scratch bound<=16GiB required')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {
        'analysis_id': 'source_readiness', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'participant': args.participant, 'session': args.session, 'complete_neural_session_handoff': False,
        'full_source_qualified': False, 'source_root': str(args.source_root),
        'payload_dir': str(args.payload_dir), 'phase': 'frozen_metadata',
        'scratch_limit_bytes': args.scratch_limit_bytes, 'internal_deadline_seconds': args.max_seconds,
        'source_volumes_consumed': 0, 'uncompressed_beta_payload_bytes_consumed': 0,
        'selected_volumes_decoded': 0, 'excluded_volumes_discarded_as_opaque_bytes': 0,
        'excluded_beta_volumes_numerically_decoded': 0, 'selected_roi_nonfinite_values': 0,
        'selected_roi_zero_values': 0, 'roi_components': [], 'errors': [],
        'raw_images_or_noise_ceiling_maps_read': False,
        'session_standardization_reliability_RDM_or_model_fit_performed': False,
        'boundary': 'Trusted selected-trial reader: gzip may decode all stream bytes, but only frozen development volumes become numeric arrays; no OS-level audit seal.',
        'numeric_validity': 'Original float32 beta zeros and nonfinite values are retained; no training voxel selection or repeat eligibility is decided by this I/O pilot.'}

    def save_progress():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output_dir / 'neural_handoff.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output_dir / 'neural_handoff.json')
        print(json.dumps({key: report[key] for key in
              ('phase', 'source_volumes_consumed', 'selected_volumes_decoded', 'elapsed_seconds')}), flush=True)

    def deadline(*_):
        raise TimeoutError('Selected neural handoff internal time bound reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        summary, arrays, beta, rows, rois = frozen_inputs(
            args.metadata_dir, args.array_metadata, args.roi_mapping, args.participant, args.session)
        if str(args.source_root) != summary['source_root'] or str(args.source_root) != arrays['source_root']:
            raise ValueError('Source root differs from the authorized frozen metadata')
        fresh_payload_directory(args.payload_dir)
        report['selected_trial_count'] = len(rows)
        report['total_trial_count'] = beta['header']['shape'][3]
        report['beta_path'] = beta['path']
        report['phase'] = 'roi_membership'
        masks, evc = [], np.zeros(math.prod(SPATIAL_SHAPE), dtype=bool)
        for item in rois:
            mask, nan_count = read_mask(args.source_root / 'dataset' / item['path'], item['header'])
            report['roi_components'].append({'label': item['label'], 'path': item['path'],
                                            'member_voxels': int(mask.sum()), 'NaNs_excluded': nan_count})
            if item['family'] == 'laion':
                masks.append(mask)
            else:
                evc |= mask
        masks.append(evc)
        union_indices = np.flatnonzero(np.logical_or.reduce(masks))
        membership = np.stack([mask[union_indices] for mask in masks], axis=1)
        report['roi_names'] = list(ROI_NAMES)
        report['roi_voxel_counts'] = {name: int(mask.sum()) for name, mask in zip(ROI_NAMES, masks)}
        report['union_voxel_count'] = len(union_indices)
        if not len(union_indices) or any(count == 0 for count in report['roi_voxel_counts'].values()):
            raise ValueError('A required fixed mask is empty; no replacement is selected')
        shape = (len(rows), len(union_indices))
        planned_bytes = 4 * math.prod(shape) + union_indices.nbytes + membership.nbytes + 1024 * 1024
        if planned_bytes > args.scratch_limit_bytes:
            raise ValueError('Selected output exceeds the remaining prospective scratch bound')
        report['planned_payload_upper_bytes'] = planned_bytes
        report['selected_beta_payload_bytes'] = 4 * math.prod(shape)
        save_progress()
        partial = args.payload_dir / 'development_betas.partial.npy'
        report['phase'] = 'selected_beta_stream'
        save_progress()
        with partial.open('xb') as output:
            destination = SequentialNpyRows(output, shape)
            with gzip.open(args.source_root / 'dataset' / beta['path'], 'rb') as stream:
                current, dtype = open_header(stream, beta['header'])
                stream_selected(stream, current['shape'], dtype, rows, union_indices,
                                destination, report, checkpoint=save_progress)
        np.save(args.payload_dir / 'voxel_indices_c.npy', union_indices, allow_pickle=False)
        np.save(args.payload_dir / 'roi_membership.npy', membership, allow_pickle=False)
        with (args.payload_dir / 'development_trials.jsonl').open('w') as target:
            for row in rows:
                target.write(json.dumps(row, sort_keys=True) + '\n')
        final_array = args.payload_dir / 'development_betas.npy'
        partial.replace(final_array)
        report['artifacts'] = {name: str(args.payload_dir / name) for name in
            ('development_betas.npy', 'voxel_indices_c.npy', 'roi_membership.npy', 'development_trials.jsonl')}
        report['phase'] = 'completed_session_handoff'
        report['complete_neural_session_handoff'] = True
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
        report['phase_at_failure'] = report['phase']
        report['phase'] = 'incomplete'
    finally:
        signal.alarm(0)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_progress()
    return 0 if report['complete_neural_session_handoff'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
