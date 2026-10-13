#!/usr/bin/env python3
"""Finite pixel-only EP18 worker; no routing, source filenames or EEG inputs.

The October 12 boundary is procedural same-account separation, not OS isolation.
One fresh process owns one frozen image family, using batch size one. Captions
and token diagnostics stay in the feature artifact; terminal output is counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import zipfile

from feature_extractors import DIMENSIONS, ImageExtractor, pixels_only, validate_ids

FAMILIES = tuple(DIMENSIONS)
PILOT_COUNT = 32
RSS_BYTES = 13 * 1024**3 // 2
BUNDLE_BYTES = 256 * 1024**2
DURABLE_BYTES = 64 * 1024**2
FAMILY_BYTES = 8 * 1024**2


class PilotContractError(ValueError):
    """Only fixed, source-independent diagnostic strings may use this type."""


def require(condition, message):
    if not condition:
        raise PilotContractError(message)


@contextmanager
def bounded_process(deadline, rss_bytes=RSS_BYTES):
    """Enforce absolute wall clock and observed historical peak RSS in-process.

    Python signals run at the next interpreter opportunity; the enclosing Slurm
    allocation is the hard memory/time boundary during a native library call.
    """
    require(math.isfinite(deadline) and deadline > time.time(), "Expired process deadline")

    def check(*_):
        if time.time() >= deadline:
            raise TimeoutError("Image pilot absolute deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > rss_bytes:
            raise MemoryError("Image pilot historical RSS allowance exceeded")

    def interrupted(*_):
        raise InterruptedError("Image pilot interrupted")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        signal.signal(signal.SIGALRM, check)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        signal.setitimer(signal.ITIMER_REAL, min(.5, deadline-time.time()), .5)
        check()
        yield check
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def write_json(path, payload, byte_cap):
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    require(len(encoded) <= byte_cap, "JSON artifact byte allowance exceeded")
    with path.open("xb") as target:
        target.write(encoded)
    return len(encoded)


def pack_ids(path, expected_count=PILOT_COUNT):
    """Inspect only the delivered opaque-ID raster pack; no source crosswalk."""
    require(path.stat().st_size <= BUNDLE_BYTES, "Pixel pack exceeds allowance")
    with zipfile.ZipFile(path) as pack:
        members = pack.infolist()
        ids = [item.filename.removesuffix(".npy") for item in members]
        validate_ids(ids, expected_count)
        require(len(ids) == expected_count and ids == sorted(ids), "Wrong ordered pilot image inventory")
        require(all(item.filename == key + ".npy" and item.compress_type == zipfile.ZIP_STORED
                    for item, key in zip(members, ids)), "Only uncompressed opaque-ID pixel arrays are allowed")
        require(sum(item.file_size for item in members) <= BUNDLE_BYTES, "Decoded pixel pack exceeds allowance")
    return ids


def extract_family(family, bundle, assets, output, check, extractor_factory=ImageExtractor,
                   expected_count=PILOT_COUNT):
    """Write all rows or no completion report; retained partials are never success."""
    import numpy as np

    require(family in FAMILIES, "Unregistered image family")
    ids = pack_ids(bundle, expected_count)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    check()
    extractor = extractor_factory(family, assets, device="cpu")
    if hasattr(extractor, "torch"):
        extractor.torch.set_num_interop_threads(1)
    load_seconds = time.monotonic() - started
    check()
    rows, details, encoding_seconds = [], [], []
    with np.load(bundle, allow_pickle=False) as pack:
        for key in ids:
            check()
            array = pixels_only(pack[key])
            before = time.monotonic()
            values, detail = extractor.encode([key], [array])
            encoding_seconds.append(time.monotonic() - before)
            require(values.dtype == np.float32 and values.shape == (1, DIMENSIONS[family])
                    and np.isfinite(values).all(), "Incomplete or nonfinite selected image features")
            rows.append(values[0].copy())
            details.append({"opaque_image_id": key, "details": detail})
            del array, values
            check()
        # Selected prospectively: one extra inference of the first image at the
        # actual singleton batch size. This makes no cross-batch parity claim.
        repeated_started = time.monotonic()
        repeated, repeated_details = extractor.encode([ids[0]], [pixels_only(pack[ids[0]])])
        repeat_seconds = time.monotonic()-repeated_started
        require(repeated.dtype == np.float32 and repeated.shape == (1, DIMENSIONS[family])
                and np.isfinite(repeated).all(), "Incomplete singleton repeatability output")
        repeat_error = float(np.max(np.abs(repeated[0]-rows[0])))
        require(np.allclose(repeated[0], rows[0], atol=1e-6, rtol=1e-5),
                "Selected singleton inference is not numerically repeatable")
        require(repeated_details == details[0]["details"],
                "Selected singleton caption/token/truncation details changed on repetition")
        del repeated, repeated_details
        check()
    features = np.stack(rows)
    require(features.nbytes + 4096 < FAMILY_BYTES, "Feature array exceeds family allowance")
    with (output / "features.npy").open("xb") as target:
        np.save(target, features, allow_pickle=False)
    used = (output / "features.npy").stat().st_size
    used += write_json(output / "image_details.json", details, FAMILY_BYTES - used - 65536)
    check()
    report = {"status": "complete_image_family", "family": family, "images": len(ids),
              "opaque_ids": ids, "shape": list(features.shape), "dtype": "float32", "batch_size": 1,
              "load_seconds": load_seconds, "encoding_seconds": encoding_seconds,
              "inferences": len(ids)+1, "unique_images": len(ids),
              "singleton_repeatability": {"opaque_image_id": ids[0], "extra_inferences": 1,
                                         "seconds": repeat_seconds, "maximum_absolute_error": repeat_error,
                                         "atol": 1e-6, "rtol": 1e-5, "details_identical": True,
                                         "cross_batch_equivalence_claimed": False},
              "elapsed_seconds": time.monotonic()-started,
              "process_cpu_seconds": time.process_time(),
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "feature_and_detail_bytes": used, "semantic_source_metadata_received": False,
              "separation": "procedural_same_account_not_OS_isolation"}
    # This report is the sole family completion marker and is written last.
    write_json(output / "report.json", report, min(65536, FAMILY_BYTES-used))
    check()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--pixel-bundle", type=Path, required=True)
    parser.add_argument("--asset-paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        from qualify_model_runtime import require_compute
        require_compute()
        require(os.environ.get("SLURM_MEM_PER_NODE") == "7168" and not os.environ.get("SLURM_JOB_GPUS"),
                "Selected pilot requires 1 CPU, 7 GiB and no GPU")
        with bounded_process(args.deadline_unix) as check:
            assets = json.loads(args.asset_paths.read_text())
            report = extract_family(args.family, args.pixel_bundle, assets, args.output_dir, check)
        print(json.dumps({key: report[key] for key in ("status", "family", "images", "elapsed_seconds", "max_rss_kib")}), flush=True)
        return 0
    except Exception as error:
        # No exception text/traceback: decoder/library errors can contain a name.
        failure = {"status": "image_family_failure", "family": args.family,
                   "error_type": type(error).__name__}
        if isinstance(error, PilotContractError):
            failure["contract_diagnostic"] = str(error)
        if args.output_dir.is_dir() and not (args.output_dir/"failure.json").exists():
            write_json(args.output_dir/"failure.json", failure, 65536)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
