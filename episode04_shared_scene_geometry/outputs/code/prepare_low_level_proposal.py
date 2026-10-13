#!/usr/bin/env python3
"""Approved fixed EP04 unique-development-only low-level extraction.

The scientist approved the exact RGB block-moment proposal on 2026-10-05.
The coordinator selects the representative/full command and finite resources;
there is no automatic continuation, new model, shared-image or OOD access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time

import h5py
import numpy as np
import PIL

from low_level_pixels_proposal import CONTROL_ID, DIMENSION, descriptor, displayed_rgb
from prepare_source_metadata import CELLS, PARTICIPANTS, SOURCE_NAMES


def bind_unique_rows(metadata_directory, raw_metadata):
    """Fully join role/identity metadata before opening an HDF5 image element."""
    report = json.loads((metadata_directory/'source_metadata_report.json').read_text())
    if report.get('metadata_fold_freeze_complete') is not True or report.get('development_images') != 23560:
        raise ValueError('Completed full unique-development metadata freeze required')
    development = {}
    with (metadata_directory/'development_images.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['image_id']
            if key in development or (row['participant'], row['source_stratum']) not in CELLS:
                raise ValueError('Duplicated identity or unknown development participant/source cell')
            if type(row['fold']) is not int or row['fold'] not in range(5):
                raise ValueError('Frozen image lacks a supported fold identity')
            development[key] = row
    if len(development) != 23560 or Counter(row['participant'] for row in development.values()) != Counter({p: 4712 for p in PARTICIPANTS}):
        raise ValueError('Development image set is not the complete selected cohort')
    bound, seen, source_unique, row_count = {}, set(), set(), 0
    with raw_metadata.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['image_name', 'dataset', 'participant', 'unique_or_shared', 'n_reps']:
            raise ValueError('Raw stimulus CSV does not have the declared five-column identity schema')
        for index, row in enumerate(reader):
            row_count += 1
            key = row['image_name']
            if not key or key in seen:
                raise ValueError('Raw source CSV has a missing/duplicate image identity')
            seen.add(key)
            if row['unique_or_shared'] == 'unique':
                source_unique.add(key)
            if key not in development:
                continue  # Never return an excluded identity or HDF row to the reader.
            declared = development[key]
            if (row['unique_or_shared'] != 'unique' or row['dataset'] not in SOURCE_NAMES
                    or SOURCE_NAMES[row['dataset']] != declared['source_stratum']
                    or row['participant'] != declared['participant']
                    or row['n_reps'] != declared['n_reps_csv']):
                raise ValueError('Raw source role/participant/repeat metadata conflicts with frozen development')
            bound[key] = {**declared, 'source_row': index}
    if row_count != 25052 or set(bound) != set(development) or source_unique != set(development):
        raise ValueError('Raw/source development coverage is incomplete or inconsistent')
    return bound, row_count


def selected_ids(bound, mode):
    if mode == 'full':
        return sorted(bound)
    if mode != 'representative':
        raise ValueError('Choose the representative or complete development pass explicitly')
    selected = []
    for participant, stratum in CELLS:
        ids = sorted(key for key, row in bound.items()
                     if row['participant'] == participant and row['source_stratum'] == stratum)
        if len(ids) < 4:
            raise ValueError('The prespecified representative lacks four images in a cell')
        selected.extend(ids[:4])
    if len(selected) != 60 or len(set(selected)) != 60:
        raise ValueError('Representative must contain sixty distinct development images')
    return selected


class DevelopmentImages:
    """Fixed approved identity-to-row map; no bulk/iterator/raw-row read API."""
    def __init__(self, path, bound, selected, expected_rows):
        self.path = path
        self.allowed = {key: bound[key]['source_row'] for key in selected}
        self.expected_rows = expected_rows
        self.handle = None

    def __enter__(self):
        self.handle = h5py.File(self.path, 'r')
        try:
            self.images = self.handle['images']
            if not isinstance(self.images, h5py.Dataset) or self.images.shape != (self.expected_rows,):
                raise ValueError('Raw images dataset does not match provider metadata row axis')
        except Exception:
            self.handle.close()
            raise
        return self

    def __exit__(self, *_):
        self.handle.close()

    def read(self, image_id):
        if image_id not in self.allowed:
            raise ValueError('Image identity is outside the selected unique-development read set')
        value = self.images[self.allowed[image_id]]  # The sole source image-element read.
        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value)
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != np.uint8:
            raise ValueError('Expected one encoded byte vector; no format fallback')
        return value.tobytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--raw-metadata', type=Path, required=True)
    parser.add_argument('--raw-h5', type=Path, required=True)
    parser.add_argument('--mode', choices=('representative', 'full'), required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use the coordinator-owned verified Slurm allocation')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '1' or any(
            os.environ.get(key) != '1' for key in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')):
        raise RuntimeError('Selected extraction requires one CPU and all six numerical thread limits equal one')
    if args.max_seconds <= 0 or args.output_dir.exists():
        raise ValueError('Positive execution bound and new output directory required')
    if args.mode == 'representative' and args.max_seconds > 300:
        raise ValueError('The approved representative is bounded to 300 seconds')
    started = time.monotonic()
    args.output_dir.mkdir(parents=True)
    report = {'control_id': CONTROL_ID, 'mode': args.mode, 'status': 'incomplete',
              'selected': 0, 'completed': 0, 'complete_development_coverage': False,
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
              'raw_h5': str(args.raw_h5), 'raw_metadata': str(args.raw_metadata),
              'metadata_dir': str(args.metadata_dir),
              'versions': {'numpy': np.__version__, 'Pillow': PIL.__version__, 'h5py': h5py.__version__},
              'stored_dtype': '<f8', 'feature_dimension': DIMENSION,
              'internal_seconds': args.max_seconds,
              'source_read_seconds': 0., 'decode_transform_seconds': 0.,
              'source_image_bytes_read': 0, 'feature_values_or_images_printed': False,
              'excluded_regular_shared_or_ood_images_read': False,
              'neural_values_read': False, 'candidate_scores_computed': False,
              'boundary': 'Trusted source-selecting reader; same UID is not OS audit isolation'}
    features = None

    def deadline(*_):
        raise TimeoutError('Selected low-level preparation bound reached')

    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()

    def publish():
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = args.output_dir/'low_level_preparation_report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output_dir/'low_level_preparation_report.json')

    previous = {sig: signal.getsignal(sig) for sig in
                (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, deadline)
        signal.alarm(args.max_seconds)
        publish()
        check()
        bound, source_rows = bind_unique_rows(args.metadata_dir, args.raw_metadata)
        identifiers = selected_ids(bound, args.mode)
        report['selected'] = len(identifiers)
        (args.output_dir/'image_ids.json').write_text(json.dumps(identifiers)+'\n')
        print(json.dumps({'phase': 'all_development_metadata_joined', 'selected': len(identifiers)}), flush=True)
        features = np.lib.format.open_memmap(args.output_dir/'low_level_features.partial.npy',
                                             mode='w+', dtype='<f8', shape=(len(identifiers), DIMENSION))
        publish()
        check()
        with DevelopmentImages(args.raw_h5, bound, identifiers, source_rows) as source:
            for index, image_id in enumerate(identifiers):
                check()
                read_started = time.monotonic()
                encoded = source.read(image_id)
                report['source_read_seconds'] += time.monotonic()-read_started
                report['source_image_bytes_read'] += len(encoded)
                check()
                transform_started = time.monotonic()
                features[index] = descriptor(displayed_rgb(encoded))
                report['decode_transform_seconds'] += time.monotonic()-transform_started
                report['completed'] += 1
                if report['completed'] % (10 if args.mode == 'representative' else 1000) == 0:
                    features.flush()
                    publish()
                    print(json.dumps({'phase': 'unique_images_transformed', 'completed': report['completed'],
                                      'selected': report['selected']}), flush=True)
                check()
        features.flush()
        del features
        features = None
        check()
        (args.output_dir/'low_level_features.partial.npy').rename(args.output_dir/'low_level_features.npy')
        report['status'] = 'complete'
        report['complete_development_coverage'] = args.mode == 'full' and report['completed'] == 23560
        publish()
        check()
    except Exception as exc:
        report['status'] = 'incomplete'
        report['complete_development_coverage'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if features is not None:
            features.flush()
        if report['status'] == 'complete' and time.monotonic()-started >= args.max_seconds:
            report['status'] = 'incomplete'
            report['complete_development_coverage'] = False
            report['error'] = 'TimeoutError: Selected bound reached during finalization'
        publish()
        if report['status'] == 'complete' and time.monotonic()-started >= args.max_seconds:
            report['status'] = 'incomplete'
            report['complete_development_coverage'] = False
            report['error'] = 'TimeoutError: Selected bound reached during final publication'
            publish()
    print(json.dumps({key: report[key] for key in ('status', 'selected', 'completed', 'elapsed_seconds')}), flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
