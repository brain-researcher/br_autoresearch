#!/usr/bin/env python3
"""One synthetic, node-local comparison of contiguous HDF5 truth reads.

Run only in an allocated compute shell. The sole project import is the isolated
truth HDF opener; no empirical inputs are accessed.
/proc/self/io rchar and syscr count logical read syscall traffic, not physical I/O.
The freshly written fixture remains in the filesystem cache; this is not a
cold-storage throughput measurement.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

for _thread_name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from truth_hdf_io import open_truth_readonly


IMAGES, CHANNELS, HEIGHT, WIDTH = 224, 512, 14, 14
SPATIAL = HEIGHT * WIDTH
UNITS = CHANNELS * SPATIAL
CHANNEL_SELECTION = (0, 127, 255, 511)
LAYER = "conv5_1"
RSS_LIMIT_KIB = 512 * 1024


def _io_counters():
    with open("/proc/self/io", encoding="ascii") as stream:
        counters = dict(line.split(":", 1) for line in stream)
    return {key: int(counters[key]) for key in ("rchar", "syscr")}


def _open_readonly(path, sieve_bytes):
    if sieve_bytes == 0:
        return open_truth_readonly(path)
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    try:
        access.set_fapl_sec2()
        access.set_sieve_buf_size(sieve_bytes)
        identifier = h5py.h5f.open(os.fsencode(path), h5py.h5f.ACC_RDONLY, fapl=access)
    finally:
        access.close()
    try:
        return h5py.File(identifier)
    except BaseException:
        identifier.close()
        raise


def _write_fixture(directory):
    features_path = directory / "features.h5"
    flags_path = directory / "finite_packed.h5"
    unit_ids = np.arange(UNITS, dtype=np.int64)
    with h5py.File(features_path, "w", driver="sec2") as features, \
            h5py.File(flags_path, "w", driver="sec2") as flags:
        data = features.create_dataset(
            LAYER, shape=(IMAGES, CHANNELS, HEIGHT, WIDTH), dtype="<f4",
            chunks=None, compression=None,
        )
        packed = flags.create_dataset(
            LAYER, shape=(IMAGES, (UNITS + 7) // 8), dtype="u1",
            chunks=None, compression=None,
        )
        # Materialize every row, avoiding sparse/unallocated fill-value reads.
        for row in range(IMAGES):
            valid = (unit_ids + 3 * row) % 97 != 0
            values = (unit_ids + row * 1024).astype(np.float32)
            values[~valid] = np.nan
            data[row] = values.reshape(CHANNELS, HEIGHT, WIDTH)
            packed[row] = np.packbits(valid, bitorder="little")
    return features_path, flags_path


def _expected(channel):
    first = channel * SPATIAL
    first_byte, stop_byte = first // 8, (first + SPATIAL + 7) // 8
    rows = np.arange(IMAGES, dtype=np.int64)[:, None]
    unit_ids = np.arange(first, first + SPATIAL, dtype=np.int64)[None, :]
    valid = (unit_ids + 3 * rows) % 97 != 0
    values = (unit_ids + rows * 1024).astype(np.float32)
    values[~valid] = np.nan
    # Include the neighbouring channel's bits sharing the edge bytes.
    bit_ids = np.arange(first_byte * 8, stop_byte * 8, dtype=np.int64)[None, :]
    packed = np.packbits((bit_ids + 3 * rows) % 97 != 0, axis=1, bitorder="little")
    return values, valid, packed, first - first_byte * 8


def _measure(paths, sieve_bytes, pair, order):
    returned = []
    with ExitStack() as stack:
        features = stack.enter_context(_open_readonly(paths[0], sieve_bytes))
        flags = stack.enter_context(_open_readonly(paths[1], sieve_bytes))
        for handle, shape in (
            (features, (IMAGES, CHANNELS, HEIGHT, WIDTH)),
            (flags, (IMAGES, (UNITS + 7) // 8)),
        ):
            access = handle.id.get_access_plist()
            try:
                if access.get_sieve_buf_size() != sieve_bytes:
                    raise RuntimeError("requested sieve setting was not applied")
            finally:
                access.close()
            dataset = handle[LAYER]
            if dataset.shape != shape or dataset.chunks is not None or dataset.compression is not None:
                raise RuntimeError("fixture must have the exact contiguous uncompressed layout")
            del dataset

        before = _io_counters()
        started = time.perf_counter()
        for channel in CHANNEL_SELECTION:
            first = channel * SPATIAL
            first_byte, stop_byte = first // 8, (first + SPATIAL + 7) // 8
            # Match TruthReader's per-channel dataset lookups and hyperslabs.
            values = features[LAYER][:, channel, :, :].reshape(IMAGES, SPATIAL)
            packed = flags[LAYER][:, first_byte:stop_byte]
            returned.append((channel, values, packed))
        elapsed = time.perf_counter() - started
        after = _io_counters()

    for channel, values, packed in returned:
        expected_values, expected_valid, expected_packed, bit_offset = _expected(channel)
        valid = np.unpackbits(packed, axis=1, bitorder="little")[:, bit_offset:bit_offset + SPATIAL].astype(bool)
        if not (
            np.array_equal(values, expected_values, equal_nan=True)
            and np.array_equal(packed, expected_packed)
            and np.array_equal(valid, expected_valid)
            and np.array_equal(valid, np.isfinite(values))
        ):
            raise AssertionError(f"synthetic value/packed-bit parity failed for channel {channel}")
    return {
        "pair": pair, "order": order, "sieve_bytes": sieve_bytes,
        "elapsed_seconds": elapsed,
        "logical_rchar_bytes": after["rchar"] - before["rchar"],
        "logical_read_syscalls": after["syscr"] - before["syscr"],
        "returned_array_bytes": sum(values.nbytes + packed.nbytes for _, values, packed in returned),
        "value_and_packed_bit_parity": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in os.uname().nodename:
        raise RuntimeError("an allocated compute shell is required")
    scratch_value = os.environ.get("L_SCRATCH_JOB")
    if not scratch_value:
        raise RuntimeError("L_SCRATCH_JOB must name this job's node-local scratch directory")
    scratch_root = Path(scratch_value).resolve(strict=True)
    if not args.work_dir.is_absolute():
        raise ValueError("--work-dir must be absolute and within L_SCRATCH_JOB")
    work_dir = args.work_dir.resolve()
    if not work_dir.is_relative_to(scratch_root):
        raise ValueError("--work-dir must be within L_SCRATCH_JOB")
    work_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="ep21_truth_sieve_", dir=work_dir) as temporary:
        fixture_directory = Path(temporary)
        paths = _write_fixture(fixture_directory)
        fixture_bytes = sum(path.stat().st_size for path in paths)
        for pair in range(3):
            settings = (65536, 0) if pair % 2 == 0 else (0, 65536)
            for order, sieve_bytes in enumerate(settings):
                records.append(_measure(paths, sieve_bytes, pair, order))
        maximum_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if maximum_rss_kib > RSS_LIMIT_KIB:
            raise RuntimeError(f"fixture exceeded 512 MiB RSS: {maximum_rss_kib} KiB")
    summary = {}
    for sieve_bytes in (65536, 0):
        selected = [record for record in records if record["sieve_bytes"] == sieve_bytes]
        summary[str(sieve_bytes)] = {
            field: statistics.median(record[field] for record in selected)
            for field in ("elapsed_seconds", "logical_rchar_bytes", "logical_read_syscalls")
        }
    default, disabled = summary["65536"], summary["0"]
    return {
        "status": "ok", "hostname": os.uname().nodename,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "h5py_version": h5py.__version__, "hdf5_version": h5py.version.hdf5_version,
        "feature_shape": [IMAGES, CHANNELS, HEIGHT, WIDTH],
        "flag_shape": [IMAGES, (UNITS + 7) // 8], "channels": CHANNEL_SELECTION,
        "fixture_file_bytes": fixture_bytes, "maximum_rss_kib": maximum_rss_kib,
        "value_and_packed_bit_parity": True,
        "fixture_removed": not fixture_directory.exists(),
        "counter_semantics": "rchar/syscr are logical syscall counters, not physical storage bytes; snapshot reads add small overhead",
        "cache_scope": "freshly written local fixture; no cold-cache or shared-storage throughput claim",
        "measurements": records, "medians_by_sieve_bytes": summary,
        "median_default_over_disabled_elapsed": default["elapsed_seconds"] / disabled["elapsed_seconds"],
        "median_default_over_disabled_rchar": default["logical_rchar_bytes"] / disabled["logical_rchar_bytes"],
    }


if __name__ == "__main__":
    try:
        report = main()
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), flush=True)
        raise SystemExit(1)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
