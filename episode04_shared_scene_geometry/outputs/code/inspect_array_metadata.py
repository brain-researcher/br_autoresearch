#!/usr/bin/env python3
"""EP04 structural array/ROI handoff only: no dataset or voxel values."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import time
import zlib


PARTICIPANTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
MODELS = {'CLIP': 1024, 'DINOv2': 1024, 'PEcore': 1024, 'SigLIP2': 1152}
SECTORS = ('laion-ventral', 'laion-lateral', 'laion-dorsal')
EVC_COMPONENTS = ('V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v')
ATTRIBUTE_WHITELIST = (
    'model', 'model_name', 'model_id', 'architecture', 'checkpoint', 'pretrained',
    'pretrained_name', 'revision', 'model_revision', 'layer', 'pooling',
    'normalize', 'normalized', 'normalization', 'image_size', 'input_size',
    'resolution', 'preprocessing', 'version', 'source',
)


def read_json(path, maximum=4_000_000):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f'Metadata exceeds the selected bound: {path.name}')
    return json.loads(raw)


def header_bytes(stream):
    """Inflate first four bytes, then exactly the identified NIfTI header."""
    decoder, compressed, result = zlib.decompressobj(16 + zlib.MAX_WBITS), b'', bytearray()
    needed, opaque_bytes = 4, 0
    while len(result) < needed:
        if not compressed:
            compressed = stream.read(min(1024, 65536 - opaque_bytes))
            opaque_bytes += len(compressed)
            if not compressed:
                raise ValueError('Truncated or over-bound compressed NIfTI header')
        result.extend(decoder.decompress(compressed, needed - len(result)))
        compressed = decoder.unconsumed_tail
        if len(result) == 4 and needed == 4:
            candidates = [struct.unpack(order + 'i', result)[0] for order in ('<', '>')]
            sizes = [size for size in candidates if size in (348, 540)]
            if len(sizes) != 1:
                raise ValueError('Unsupported NIfTI header size/endian')
            needed = sizes[0]
        elif decoder.eof and len(result) < needed:
            raise ValueError('Compressed stream ends before the header')
    # No flush, full gzip read, image constructor or numeric array reader.
    return bytes(result), opaque_bytes


def parse_header(raw):
    order = '<' if struct.unpack('<i', raw[:4])[0] == len(raw) else '>'
    def values(offset, fmt):
        return list(struct.unpack_from(order + fmt, raw, offset))
    if len(raw) == 348:
        if raw[344:348] != b'n+1\0':
            raise ValueError('Expected single-file NIfTI-1')
        dims, pixdim = values(40, '8h'), values(76, '8f')
        dtype, bitpix = values(70, '2h')
        offset = values(108, 'f')[0]
        slope, intercept = values(112, '2f')
        qform, sform = values(252, '2h')
        quaternion = values(256, '6f')
        rows = values(280, '12f')
        units = raw[123]
    elif len(raw) == 540:
        if raw[4:12] != b'n+2\0\r\n\x1a\n':
            raise ValueError('Expected single-file NIfTI-2')
        dims, pixdim = values(16, '8q'), values(104, '8d')
        dtype, bitpix = values(12, '2h')
        offset = values(168, 'q')[0]
        slope, intercept = values(176, '2d')
        qform, sform = values(344, '2i')
        quaternion = values(352, '6d')
        rows = values(400, '12d')
        units = values(500, 'i')[0]
    else:
        raise ValueError('Unsupported header byte count')
    if (not 1 <= dims[0] <= 7 or any(d <= 0 for d in dims[1:dims[0]+1])
            or not all(math.isfinite(v) for v in pixdim + quaternion + rows + [offset])
            or offset < len(raw)):
        raise ValueError('Invalid structural NIfTI dimensions/grid/offset')
    shape = dims[1:dims[0]+1]
    return {'shape': shape, 'datatype_code': dtype, 'bits_per_voxel': bitpix,
            'voxel_offset': offset, 'header_bytes_decompressed': len(raw),
            'scl_slope': slope if math.isfinite(slope) else None,
            'scl_inter': intercept if math.isfinite(intercept) else None,
            'spatial_grid_fields': {'shape': shape[:3], 'pixdim': pixdim[:4],
                'spatial_units_code': units & 7, 'qform_code': qform, 'sform_code': sform,
                'quaternion_and_offsets': quaternion, 'sform_rows': rows},
            'voxel_values_decoded': False}


def nifti_header(path):
    with path.open('rb') as stream:
        raw, compressed_count = header_bytes(stream)
    return parse_header(raw) | {'opaque_compressed_bytes_read': compressed_count}


def require_trial_axis(header, rows):
    if len(header['shape']) != 4 or header['shape'][3] != rows:
        raise ValueError('Session beta fourth axis differs from frozen original TSV row count')


def small_attributes(obj):
    import numpy as np
    result = {}
    for name in ATTRIBUTE_WHITELIST:
        if name not in obj.attrs:
            continue
        info = obj.attrs.get_id(name)
        try:
            shape = info.shape
            count = math.prod(shape) if shape else 1
            if count > 16 or info.get_storage_size() > 8192:
                result[name] = {'status': 'outside_small_metadata_bound', 'shape': list(shape)}
                continue
            if info.dtype.kind not in 'biufSUO':
                result[name] = {'status': 'unsupported_metadata_type', 'dtype': str(info.dtype)}
                continue
        finally:
            info.close()
        value = obj.attrs[name]
        def plain(item):
            if isinstance(item, bytes):
                return item.decode('utf-8')
            if isinstance(item, np.ndarray):
                return plain(item.tolist())
            if isinstance(item, (list, tuple)):
                return [plain(v) for v in item]
            if isinstance(item, np.generic):
                return plain(item.item())
            if item is None or isinstance(item, (str, bool, int, float)):
                return item
            raise ValueError('Nonliteral whitelisted model metadata')
        value = plain(value)
        encoded = json.dumps(value, allow_nan=False)
        result[name] = value if len(encoded.encode()) <= 8192 else {'status': 'outside_small_metadata_bound'}
    return result


def embedding_metadata(path, feature_count):
    import h5py
    with h5py.File(path, 'r') as source:
        arrays = {}
        for name in ('embedding', 'image_ids', 'valid'):
            if not isinstance(source.get(name, getlink=True), h5py.HardLink):
                raise ValueError('Expected local fixed HDF5 dataset link')
            dataset = source[name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError('Expected named HDF5 dataset')
            arrays[name] = {'shape': list(dataset.shape), 'dtype': str(dataset.dtype)}
        if (arrays['embedding']['shape'] != [25052, feature_count]
                or arrays['image_ids']['shape'] != [25052] or arrays['valid']['shape'] != [25052]):
            raise ValueError('Released embedding axes differ from the fixed documented panel')
        return {'datasets': arrays, 'file_attributes': small_attributes(source),
                'embedding_attributes': small_attributes(source['embedding']),
                'dataset_elements_read': 0, 'unique_id_coverage_verified': False,
                'validity_mask_values_read': False, 'normalization_numerically_verified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--metadata-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prior-beta-report', type=Path)
    p.add_argument('--max-seconds', type=int, default=300)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    if args.output_dir.exists() or not 0 < args.max_seconds <= 300:
        raise ValueError('Use a fresh output path and bounded positive deadline <=300s')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {'analysis_id': 'source_readiness', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(args.source_root),
        'structural_pass_complete': False, 'full_source_qualified': False,
        'neural_embedding_ROI_or_image_values_read': False, 'audit_opened': False,
        'betas': [], 'embeddings': [], 'rois': [], 'errors': [],
        'retinotopic_EVC_definition': 'Union of V1d,V1v,V2d,V2v,V3d,V3v; never laion-EVC',
        'ROI_definition_reference': 'https://laion-fmri.hebartlab.com/rois.html',
        'attribute_whitelist': list(ATTRIBUTE_WHITELIST)}
    def deadline(*_):
        raise TimeoutError('Selected structural metadata time bound reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        frozen = read_json(args.metadata_dir/'source_metadata_report.json')
        files = frozen['trial_files']
        expected = {(pid, f'ses-{n:02d}') for pid in PARTICIPANTS for n in range(1, 31)}
        if (not frozen['metadata_fold_freeze_complete'] or len(files) != 150
                or {(r['participant'], r['session']) for r in files} != expected):
            raise ValueError('Complete frozen150-session trial metadata required')
        selected = read_json(args.source_root/'provider_manifests/selected_objects.json')
        receipt_counts = Counter(r['Key'] for r in selected)
        receipt = {r['Key']: r for r in selected}
        upstream = {r['Key'] for r in read_json(args.source_root/'provider_manifests/rois.json')['Contents']}
        def locate(key):
            path = args.source_root/'dataset'/key
            if receipt_counts[key] != 1 or not path.is_file():
                raise ValueError(f'Exact selected source file absent or duplicated: {key}')
            return path
        grids = {}
        rows_to_read = files
        if args.prior_beta_report is not None:
            prior = read_json(args.prior_beta_report)
            previous = prior['betas']
            expected_rows = {(r['participant'], r['session']): r['rows'] for r in files}
            if (prior['neural_embedding_ROI_or_image_values_read'] is not False
                    or len(previous) != 150
                    or {(r['participant'], r['session']) for r in previous} != expected):
                raise ValueError('Prior beta evidence must contain the exact150 untouched-value sessions')
            for item in previous:
                pid, ses = item['participant'], item['session']
                exact_path = (f'derivatives/glmsingle-tedana/{pid}/{ses}/func/'
                    f'{pid}_{ses}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz')
                header, sidecar = item['header'], item['sidecar']
                rows = expected_rows[(pid, ses)]
                if (item['path'] != exact_path or item['original_tsv_rows'] != rows
                        or item['fourth_axis_equals_tsv_rows'] is not True
                        or header['voxel_values_decoded'] is not False
                        or header['header_bytes_decompressed'] not in (348, 540)
                        or sidecar.get('GLMsingleField') != 'betasmd'
                        or sidecar.get('ModelStage') != 'TYPED'):
                    raise ValueError('Prior beta record disagrees with the exact source/axis contract')
                require_trial_axis(header, rows)
                grids.setdefault(pid, header['spatial_grid_fields'])
            report['betas'] = previous
            report['prior_beta_evidence'] = str(args.prior_beta_report)
            report['beta_headers_reused_without_source_reread'] = 150
            rows_to_read = []
        for row in rows_to_read:
            pid, ses = row['participant'], row['session']
            base = (f'derivatives/glmsingle-tedana/{pid}/{ses}/func/'
                    f'{pid}_{ses}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap')
            header = nifti_header(locate(base+'.nii.gz'))
            require_trial_axis(header, row['rows'])
            sidecar = read_json(locate(base+'.json'), 65536)
            if sidecar.get('GLMsingleField') != 'betasmd' or sidecar.get('ModelStage') != 'TYPED':
                raise ValueError('Beta sidecar does not declare the fixed TYPED/betasmd representation')
            grids.setdefault(pid, header['spatial_grid_fields'])
            report['betas'].append({'participant': pid, 'session': ses, 'path': base+'.nii.gz',
                'original_tsv_rows': row['rows'], 'fourth_axis_equals_tsv_rows': True,
                'header': header, 'exact_spatial_fields_equal_first_session': header['spatial_grid_fields'] == grids[pid],
                'sidecar': {k: sidecar.get(k) for k in ('Description', 'Units', 'GLMsingleField', 'ModelStage', 'Sources')}})
            if len(report['betas']) % 30 == 0:
                print(json.dumps({'phase': 'beta_headers', 'completed': len(report['betas'])}), flush=True)
        for model, dimensions in MODELS.items():
            key = f'stimuli/task-images_desc-{model}_embeddings.h5'
            report['embeddings'].append({'model_label': model, 'path': key,
                'acquired_LastModified': receipt.get(key, {}).get('LastModified'),
                **embedding_metadata(locate(key), dimensions)})
        for pid in PARTICIPANTS:
            for family, labels in (('laion', SECTORS), ('retinotopy', EVC_COMPONENTS)):
                for label in labels:
                    key = f'derivatives/rois/{pid}/{family}/{pid}_space-T1w_res-1pt8_label-{label}_mask.nii.gz'
                    item = {'participant': pid, 'family': family, 'label': label, 'path': key,
                        'listed_in_preserved_provider_index': key in upstream,
                        'selected_in_completed_acquisition': receipt_counts[key] == 1}
                    if receipt_counts[key] == 0:
                        item['status'] = 'not_selected_in_acquisition_not_a_claim_of_provider_unavailability'
                    else:
                        header = nifti_header(locate(key))
                        if len(header['shape']) != 3:
                            raise ValueError('ROI header is not3D')
                        item.update(status='header_read_values_closed', header=header,
                            exact_spatial_fields_equal_beta=header['spatial_grid_fields'] == grids[pid])
                    report['rois'].append(item)
        report['structural_pass_complete'] = True
        report['remaining_limits'] = ['No array element finiteness or neural repeat validity checked',
            'No feature image IDs, valid masks or numerical normalization read',
            'No ROI union, coverage or voxel overlap computed',
            'Header equality is structural and does not establish role isolation or full source qualification']
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (args.output_dir/'array_metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'structural_pass_complete': report['structural_pass_complete'],
        'beta_headers': len(report['betas']), 'embedding_files': len(report['embeddings']),
        'ROI_metadata_rows': len(report['rois']), 'errors': len(report['errors']),
        'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return 0 if report['structural_pass_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
