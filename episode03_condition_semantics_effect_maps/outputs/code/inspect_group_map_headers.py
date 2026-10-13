#!/usr/bin/env python3
"""Read NIfTI headers for explicitly censused development maps; no voxel decoding."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import struct

from qualify_source_metadata import require_compute


def header(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read(352)
    if len(data) < 348:
        raise ValueError("Truncated NIfTI header")
    endian = "<" if struct.unpack_from("<i", data)[0] == 348 else ">"
    if struct.unpack_from(endian + "i", data)[0] != 348:
        raise ValueError("Not a supported NIfTI-1 header")
    unpack = lambda fmt, at: list(struct.unpack_from(endian + fmt, data, at))
    return {"returned_uncompressed_header_bytes": len(data), "endian": endian,
            "magic": data[344:348].decode("ascii"), "dimensions": unpack("8h", 40),
            "intent_code": unpack("h", 68)[0], "datatype_code": unpack("h", 70)[0],
            "bitpix": unpack("h", 72)[0], "pixdim": unpack("8f", 76),
            "xyzt_units": data[123], "qform_code": unpack("h", 252)[0],
            "sform_code": unpack("h", 254)[0], "qform_parameters": unpack("6f", 256),
            "sform_rows": unpack("12f", 280), "voxel_array_decoded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    require_compute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    census = json.loads(args.census.read_text())
    records, errors, seen = [], [], set()
    allowed = Path(os.environ["OAK"]) / "data/OpenNeuro_analyses/openneuro_fitlins/analyses"
    for task in census["tasks"]:
        for item in task["map_filename_candidates"]:
            path = Path(item["path"])
            if path in seen:
                continue
            seen.add(path)
            if not path.is_relative_to(allowed):
                raise ValueError("Census path leaves the authorized source root")
            try:
                records.append({"path": str(path), "dataset": task["dataset_id"],
                                "task": task["task"], "contrast": item["contrast"], **header(path)})
            except (OSError, ValueError, UnicodeError, struct.error) as exc:
                errors.append({"path": str(path), "error": str(exc), "error_type": type(exc).__name__})
    result = {"analysis_id": "source_handoff_qualification", "job_id": os.environ["SLURM_JOB_ID"],
              "status": "header_census_only_not_scientific_map_eligibility", "census": str(args.census),
              "headers_read": len(records), "errors": errors,
              "distinct_dimensions": dict(Counter(str(x["dimensions"]) for x in records)),
              "intent_codes": dict(Counter(x["intent_code"] for x in records)),
              "voxel_arrays_decoded": False, "canonical_orientation_assigned": False,
              "common_space_or_mask_selected": False, "candidate_handoff_qualified": False,
              "headers": records}
    (args.output_dir / "group_map_headers.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "headers"}, indent=2))


if __name__ == "__main__":
    main()
