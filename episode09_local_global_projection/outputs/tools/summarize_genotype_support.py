#!/usr/bin/env python3
"""Describe non-outcome metadata support; do not assign the scientific cohort."""
from collections import Counter
import json
import os
from pathlib import Path


if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Slurm compute required")
root = Path("outputs/runs/metadata_feasibility/attempt002")
report = json.loads((root / "brain_genotype_join.json").read_text())
rows = report["joined"]
groups = []
for genotype in sorted({x["genotype"] for x in rows}, key=str):
    local = [x for x in rows if x["genotype"] == genotype]
    layers = Counter()
    for x in local:
        for cell in x["source_layer_counts"]:
            layers[str(cell["layer"])] += cell["cells"]
    groups.append({"genotype": genotype, "brain_count": len(local),
                   "animal_count": len({x["animal_id"] for x in local}),
                   "target_descriptions": sorted({x["target_description"] for x in local}, key=str),
                   "layer_counts": dict(layers)})
result = {"job_id": os.environ["SLURM_JOB_ID"], "matched_brains": len(rows),
          "matched_animals": len({x["animal_id"] for x in rows}), "groups": groups,
          "unmatched": report["unmatched"], "eligibility_or_roles_assigned": False,
          "minimum_development_plus_audit_groups": 20}
(root / "genotype_support_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
