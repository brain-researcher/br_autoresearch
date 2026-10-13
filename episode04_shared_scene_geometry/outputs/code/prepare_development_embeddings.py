#!/usr/bin/env python3
"""Trusted EP04 released-feature handoff: decode only frozen development rows."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time

import h5py
import numpy as np

from prepare_source_metadata import PARTICIPANTS, CELLS


MODELS = {'CLIP': 1024, 'DINOv2': 1024, 'PEcore': 1024, 'SigLIP2': 1152}
CHUNK_ROWS = 512


def development_rows(directory):
    summary = json.loads((directory/'source_metadata_report.json').read_text())
    if summary.get('metadata_fold_freeze_complete') is not True or summary.get('development_images') != 23560:
        raise ValueError('Complete frozen development metadata required')
    rows = [json.loads(line) for line in (directory/'development_images.jsonl').read_text().splitlines()]
    if (len(rows) != 23560 or len({r['image_id'] for r in rows}) != 23560
            or Counter(r['participant'] for r in rows) != Counter({p: 4712 for p in PARTICIPANTS})
            or any((r['participant'], r['source_stratum']) not in CELLS
                   or type(r['fold']) is not int or r['fold'] not in range(5) for r in rows)):
        raise ValueError('Development identity/cohort/fold metadata disagree')
    # This stable row order is shared by all four arrays, without reassigning folds.
    return sorted(rows, key=lambda row: row['image_id'])


def identity_join(source_ids, development_ids):
    """Return source row per frozen output ID; no excluded identity is exported."""
    allowed = set(development_ids)
    if len(allowed) != len(development_ids):
        raise ValueError('Duplicate requested development identity')
    found = {}
    for index, key in enumerate(source_ids):
        if key not in allowed:
            continue
        if key in found:
            raise ValueError('Duplicate development identity in released image_ids')
        found[key] = index
    if set(found) != allowed:
        raise ValueError(f'Released feature identity coverage missing {len(allowed-set(found))} development IDs')
    return np.asarray([found[key] for key in development_ids], dtype=np.int64)


def copy_selected(features, valid, source_rows, destination, progress, chunk_rows=CHUNK_ROWS):
    """Only explicit ascending selected indices reach either value dataset."""
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError('Development source rows are not one-to-one')
    order = np.argsort(source_rows)
    for start in range(0, len(order), chunk_rows):
        output_rows = order[start:start+chunk_rows]
        selected_rows = source_rows[output_rows]
        flags = valid[selected_rows]  # Only development validity elements.
        if flags.shape != (len(selected_rows),) or flags.dtype != np.dtype(bool) or not np.all(flags):
            raise ValueError('Development validity mask is incomplete/false; no fold shrinkage')
        values = features[selected_rows, :]  # The sole feature-value read.
        if (values.shape != (len(selected_rows), destination.shape[1])
                or values.dtype != destination.dtype):
            raise ValueError('Selected feature shape/dtype changed')
        if not np.isfinite(values).all() or not np.all(np.any(values != 0, axis=1)):
            raise ValueError('Development feature has nonfinite/zero vector; no fold shrinkage')
        destination[output_rows] = values  # Original float16 values, no normalization.
        progress['completed_rows'] += len(selected_rows)
        progress['logical_selected_feature_bytes'] += int(values.nbytes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--metadata-dir', type=Path, required=True)
    p.add_argument('--array-metadata', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=270)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute required')
    if a.output_dir.exists() or not 0 < a.max_seconds <= 270:
        raise ValueError('Use a fresh output and bounded deadline<=270s')
    a.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {'analysis_id': 'source_readiness', 'job_id': os.environ['SLURM_JOB_ID'],
        'hostname': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'complete_development_embedding_handoff': False, 'full_source_qualified': False,
        'source_root': str(a.source_root), 'frozen_metadata': str(a.metadata_dir),
        'array_metadata': str(a.array_metadata), 'models': [], 'chunk_rows': CHUNK_ROWS,
        'excluded_feature_or_validity_elements_returned_to_python': 0,
        'all_image_id_metadata_read_for_join_only': True, 'OOD_categories_parsed_or_exported': False,
        'neural_ROI_or_raw_pixel_values_read': False, 'RDM_or_model_fitting': False,
        'downstream_geometry': 'Convert released values to float64, fixed row L2 normalization and cosine distance; not performed by this handoff',
        'boundary': 'Trusted selected-row reader; candidate handoff contains development-only values. Same-UID source permissions are not an OS audit seal.',
        'HDF_storage_boundary': 'HDF5 may read/decompress a storage chunk containing excluded rows internally. Only explicitly selected development feature/validity elements are returned to Python; no claim of physical-byte or codec-level exclusion.',
        'errors': []}
    def deadline(*_):
        raise TimeoutError('Selected feature-handoff deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.max_seconds)
    try:
        rows = development_rows(a.metadata_dir)
        ids = [r['image_id'] for r in rows]
        metadata = json.loads(a.array_metadata.read_text())
        if not metadata['structural_pass_complete'] or metadata['source_root'] != str(a.source_root):
            raise ValueError('Use the completed structural metadata for this exact source')
        qualified = {r['model_label']: r for r in metadata['embeddings']}
        if set(qualified) != set(MODELS):
            raise ValueError('Fixed four-model structural evidence required')
        with ExitStack() as stack:
            sources, joins = {}, {}
            # Complete all four identity joins before any feature/validity read.
            for model, dimension in MODELS.items():
                relative = f'stimuli/task-images_desc-{model}_embeddings.h5'
                if qualified[model]['path'] != relative:
                    raise ValueError('Structural model path differs from fixed release file')
                source = stack.enter_context(h5py.File(a.source_root/'dataset'/relative, 'r'))
                for name in ('embedding', 'valid', 'image_ids'):
                    if not isinstance(source.get(name, getlink=True), h5py.HardLink) or not isinstance(source[name], h5py.Dataset):
                        raise ValueError('Expected local released dataset handles')
                features, valid, identifiers = source['embedding'], source['valid'], source['image_ids']
                if (features.shape != (25052, dimension) or features.dtype.kind != 'f'
                        or features.dtype.itemsize != 2 or valid.shape != (25052,)
                        or valid.dtype != np.dtype(bool) or identifiers.shape != (25052,)):
                    raise ValueError('Released feature/identity/validity schema changed')
                joins[model] = identity_join(identifiers.asstr()[...], ids)
                sources[model] = features, valid
            with (a.output_dir/'development_rows.jsonl').open('w') as stream:
                for index, row in enumerate(rows):
                    item = {key: row[key] for key in ('image_id', 'participant', 'source_stratum', 'fold', 'component_id')}
                    item.update(output_row=index, source_rows={m: int(joins[m][index]) for m in MODELS})
                    stream.write(json.dumps(item, sort_keys=True)+'\n')
            for model, dimension in MODELS.items():
                features, valid = sources[model]
                partial = a.output_dir/f'{model}.partial.npy'
                destination = np.lib.format.open_memmap(partial, mode='w+', dtype=features.dtype,
                                                        shape=(len(ids), dimension))
                item = {'model_label': model, 'shape': [len(ids), dimension],
                    'dtype': str(features.dtype), 'output': f'{model}.npy', 'completed_rows': 0,
                    'logical_selected_feature_bytes': 0, 'source_declarations': qualified[model]['file_attributes'],
                    'HDF_storage_chunks': list(features.chunks) if features.chunks else None,
                    'HDF_compression': features.compression, 'normalization_applied': False}
                report['models'].append(item)
                try:
                    copy_selected(features, valid, joins[model], destination, item)
                    destination.flush()
                finally:
                    del destination
                print(json.dumps({'phase': 'development_embedding', 'model': model,
                                  'completed_rows': item['completed_rows']}), flush=True)
        # A partially completed panel never has the complete-handoff flag.
        for model in MODELS:
            (a.output_dir/f'{model}.partial.npy').rename(a.output_dir/f'{model}.npy')
        report['complete_development_embedding_handoff'] = True
        report['development_rows'] = len(ids)
        report['row_identity_file'] = 'development_rows.jsonl'
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/'embedding_handoff.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'complete_development_embedding_handoff': report['complete_development_embedding_handoff'],
        'models': len(report['models']), 'elapsed_seconds': report['elapsed_seconds'],
        'errors': len(report['errors'])}), flush=True)
    return 0 if report['complete_development_embedding_handoff'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
