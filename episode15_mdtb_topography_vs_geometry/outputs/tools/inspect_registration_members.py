#!/usr/bin/env python3
"""Inspect only directory entries relevant to the common registration route."""
import json
import os
from pathlib import Path
import socket
import zipfile

if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
    raise RuntimeError("Use an actual Slurm compute shell")

source = Path(os.environ["OAK"]) / "data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0/source"
report = {"numeric_members_read": False, "participants": []}
for participant in ("sub-02", "sub-04"):
    archive = source / f"MDTB_{participant}.zip"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    permitted_names = [name for name in names if any(fragment in name for fragment in
                       ("/anat/", "/suit/", "reginfo", "regressors"))]
    report["participants"].append({"participant": participant, "members": permitted_names})
output = Path("outputs/runs/source_qualification/attempt001/registration_members.json")
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
