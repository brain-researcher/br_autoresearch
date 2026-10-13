#!/usr/bin/env python3
"""Read only NIfTI headers of anatomical registration inputs in fixed archives."""
import csv
import json
import math
import os
from pathlib import Path
import socket
import struct
import zipfile


def nifti_header(stream):
    header = stream.read(348)
    if len(header) != 348:
        raise ValueError("Truncated NIfTI-1 header")
    endian = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
    if struct.unpack_from(endian + "i", header)[0] != 348:
        raise ValueError("Not a NIfTI-1 header")
    if header[344:348] != b"n+1\x00":
        raise ValueError("Expected a single-file NIfTI-1 image")
    dims = struct.unpack_from(endian + "8h", header, 40)
    if not 1 <= dims[0] <= 7 or any(x <= 0 for x in dims[1:1 + dims[0]]):
        raise ValueError("Invalid NIfTI dimensions")
    slope, intercept = struct.unpack_from(endian + "2f", header, 112)
    return {
        "shape": list(dims[1:1 + dims[0]]),
        "datatype": struct.unpack_from(endian + "h", header, 70)[0],
        "pixdim": list(struct.unpack_from(endian + "8f", header, 76)),
        "voxel_offset": struct.unpack_from(endian + "f", header, 108)[0],
        "scl_slope": slope if math.isfinite(slope) else None,
        "scl_inter": intercept if math.isfinite(intercept) else None,
        "scaling_disabled": not math.isfinite(slope) or slope == 0,
        "xyzt_units": header[123],
        "intent_code": struct.unpack_from(endian + "h", header, 68)[0],
        "qform_code": struct.unpack_from(endian + "h", header, 252)[0],
        "sform_code": struct.unpack_from(endian + "h", header, 254)[0],
        "quaternion_and_offsets": list(struct.unpack_from(endian + "6f", header, 256)),
        "sform_rows": [list(struct.unpack_from(endian + "4f", header, offset))
                       for offset in (280, 296, 312)],
        "voxel_elements_read": False,
    }


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use an actual Slurm compute shell")
    source = Path(os.environ["OAK"]) / "data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0/source"
    with (source / "participants.tsv").open() as stream:
        participants = list(csv.DictReader(stream, delimiter="\t"))
    if len(participants) != 24 or len({p["participant_id"] for p in participants}) != 24:
        raise ValueError("Expected all 24 unique participants")
    output = Path("outputs/runs/source_qualification/attempt001/anatomical_headers.json")
    if output.exists():
        raise FileExistsError(output)
    report = {"participants": [], "Task_B_numeric_members_read": False,
              "all_voxel_payloads_closed": True, "registration_route_not_yet_numerically_qualified": True}
    suffixes = ("space-SUIT_xfm.nii", "desc-cereb_mask.nii", "label-GMc_probseg.nii", "T1w.nii")
    for person in participants:
        pid = person["participant_id"]
        item = {"participant_id": pid, "headers": {}}
        with zipfile.ZipFile(source / f"MDTB_{pid}.zip") as archive:
            for suffix in suffixes:
                name = f"{pid}/anat/{pid}_{suffix}"
                with archive.open(name) as stream:
                    item["headers"][suffix] = nifti_header(stream)
            item["inverse_present"] = f"{pid}/anat/{pid}_space-SUIT_inv_xfm.nii" in archive.namelist()
            item["overlap_present"] = f"{pid}/anat/{pid}_desc-overlap_mask.nii" in archive.namelist()
        report["participants"].append(item)
    grid_keys = ("shape", "pixdim", "sform_code", "sform_rows", "qform_code", "quaternion_and_offsets")
    grids = [{key: p["headers"]["space-SUIT_xfm.nii"][key] for key in grid_keys}
             for p in report["participants"]]
    report["forward_transform_headers_share_grid"] = all(g == grids[0] for g in grids)
    report["missing_inverse"] = [p["participant_id"] for p in report["participants"] if not p["inverse_present"]]
    report["missing_overlap"] = [p["participant_id"] for p in report["participants"] if not p["overlap_present"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"participants": len(participants), "common_grid": report["forward_transform_headers_share_grid"],
                      "missing_inverse": report["missing_inverse"], "voxel_elements_read": False}))


if __name__ == "__main__":
    main()
