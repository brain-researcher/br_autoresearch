"""Bounded read-only failure diagnosis; no pixels, inference or partial reuse."""
import json
import math
import os
from pathlib import Path
import resource
import socket
import time
import zipfile

import numpy as np

assert os.environ.get("SLURM_JOB_ID") == "47579378"
assert "-ln" not in socket.gethostname()
assert os.environ.get("SLURM_CPUS_PER_TASK") == "1"
started = time.monotonic()
episode = Path(os.environ["OAK"])/"users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
config = json.loads((episode/"outputs/config/full_image_extraction_20261012_001.json").read_text())
manifest = json.loads((Path(config["pixel_cache_dir"])/"manifest.json").read_text())
task = manifest["tasks"][5]
assert task == {"task_index": 5, "family": "clip_vit_b32_image", "family_shard_index": 1,
                "start": 11224, "stop": 22448}
output = Path(config["run_dir"])/"tasks/task_00005"
assert not (output/"report.json").exists()
matrix = np.load(output/"features.partial.npy", mmap_mode="r", allow_pickle=False)
assert matrix.shape == (11224, 512) and matrix.dtype == np.float32
counts = dict(rows=0, all_zero=0, finite_nonzero=0, nonfinite=0, approximately_unit_norm=0,
              leading_finite_nonzero_rows=0)
prefix_open = True
for offset in range(0, len(matrix), 128):
    assert time.monotonic()-started < 55
    block = np.asarray(matrix[offset:offset+128])
    finite = np.isfinite(block).all(axis=1)
    nonzero = (block != 0).any(axis=1)
    norms = np.sqrt(np.sum(block.astype(np.float64)**2, axis=1))
    counts["rows"] += len(block)
    counts["all_zero"] += int(np.count_nonzero(~nonzero))
    counts["finite_nonzero"] += int(np.count_nonzero(finite & nonzero))
    counts["nonfinite"] += int(np.count_nonzero(~finite))
    counts["approximately_unit_norm"] += int(np.count_nonzero(finite & (np.abs(norms-1) <= 1e-4)))
    for good in finite & nonzero:
        if prefix_open and good:
            counts["leading_finite_nonzero_rows"] += 1
        else:
            prefix_open = False
del matrix

# Only NPY headers for the first32 assigned images; never decode pixel arrays.
headers = []
for pack in manifest["packs"]:
    first, last = max(task["start"], pack["start"]), min(task["start"]+32, pack["stop"])
    if first >= last:
        continue
    path = Path(config["pixel_cache_dir"])/pack["name"]
    assert path.stat().st_size == pack["bytes"] <= config["pixel_pack_byte_cap"]
    with zipfile.ZipFile(path) as archive:
        for index in range(first, last):
            assert time.monotonic()-started < 55
            member = archive.getinfo(manifest["opaque_ids"][index]+".npy")
            assert member.compress_type == zipfile.ZIP_STORED
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                assert version in ((1, 0), (2, 0))
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(source, max_header_size=65536)
                assert len(shape) == 3 and shape[2] == 3 and min(shape) > 0
                assert dtype == np.dtype("uint8") and not fortran
                headers.append((shape, math.prod(shape)))
assert len(headers) == 32
print(json.dumps({"status": "complete_read_only_task005_diagnosis", "task_index": 5,
    "partial_rows": counts, "partial_rows_accepted_or_reused": False,
    "details_partial_bytes": (output/"image_details.partial.jsonl").stat().st_size,
    "pixel_headers_only": {"count": len(headers), "first_shape": headers[0][0],
        "first_decoded_bytes": headers[0][1], "maximum_decoded_bytes": max(v[1] for v in headers),
        "maximum_side": max(max(v[0][:2]) for v in headers)},
    "pixel_values_read": False, "model_inference": False, "EEG_or_audit_read": False,
    "elapsed_seconds": time.monotonic()-started,
    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
