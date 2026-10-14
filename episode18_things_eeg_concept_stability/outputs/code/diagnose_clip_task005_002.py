"""Read only image-array headers around the diagnosed partial-write boundary."""
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
assert "-ln" not in socket.gethostname() and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
started = time.monotonic()
episode = Path(os.environ["OAK"])/"users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
prior = json.loads((episode/"outputs/runs/feature_preparation/image_task005_diagnostic001/report.json").read_text())
assert prior["status"] == "complete_read_only_task005_diagnosis"
prefix = prior["partial_rows"]["leading_finite_nonzero_rows"]
assert prefix == 7638 and prior["partial_rows"]["approximately_unit_norm"] == prefix
config = json.loads((episode/"outputs/config/full_image_extraction_20261012_001.json").read_text())
manifest = json.loads((Path(config["pixel_cache_dir"])/"manifest.json").read_text())
boundary = manifest["tasks"][5]["start"]+prefix
assert boundary == 18862
records = []
for pack in manifest["packs"]:
    first, last = max(boundary-16, pack["start"]), min(boundary+17, pack["stop"])
    if first >= last:
        continue
    path = Path(config["pixel_cache_dir"])/pack["name"]
    assert path.stat().st_size == pack["bytes"] <= config["pixel_pack_byte_cap"]
    with zipfile.ZipFile(path) as archive:
        for index in range(first, last):
            assert time.monotonic()-started < 55
            info = archive.getinfo(manifest["opaque_ids"][index]+".npy")
            assert info.compress_type == zipfile.ZIP_STORED
            with archive.open(info) as source:
                version = np.lib.format.read_magic(source)
                assert version in ((1, 0), (2, 0))
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(source, max_header_size=65536)
            assert len(shape) == 3 and shape[2] == 3 and min(shape) > 0
            assert dtype == np.dtype("uint8") and not fortran
            records.append({"offset_from_first_zero_row": index-boundary, "shape": shape,
                            "decoded_bytes": math.prod(shape), "aspect_ratio": max(shape[:2])/min(shape[:2])})
assert len(records) == 33
print(json.dumps({"status": "complete_boundary_header_diagnosis", "task_index": 5,
    "headers": records, "pixel_values_read": False, "model_inference": False,
    "partial_rows_accepted_or_reused": False, "elapsed_seconds": time.monotonic()-started,
    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
