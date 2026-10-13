#!/usr/bin/env python3
"""Observe one public brain/animal/image alias; never modify a join or assign roles.

Only the retained public brain-level metadata workbook and existing non-outcome
EP09 join are read. No source-cell workbook, morphology, axon, or sibling output.
"""
import json
import os
from pathlib import Path
import re
import socket

from inspect_public_brain_metadata import workbook_rows


ROOT = Path(__file__).resolve().parents[1] / "runs/metadata_feasibility/attempt002"
SELECTED_COLUMNS = {
    "A": "animal_id", "B": "image_id", "C": "technique", "J": "genotype",
    "K": "genotype_provenance", "L": "target_description",
    "M": "target_reference", "N": "source_url",
}


def main():
    host = socket.gethostname()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job) or "-ln" in host or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("actual one-CPU Slurm compute shell required")
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("explicit single-thread numerical environment required")

    source = ROOT / "provider_brain_metadata.xlsx"
    if source.stat().st_size != 93468:
        raise RuntimeError("retained public metadata size changed; reconcile provenance first")
    rows = workbook_rows(source)["xl/worksheets/sheet1.xml"]
    if rows[0].get("A1") != "Animal ID" or rows[0].get("B1") != "Image ID":
        raise RuntimeError("explicit Animal ID and Image ID namespaces required")
    selected = []
    for row in rows[1:]:
        values = {re.sub(r"[0-9]", "", key): value for key, value in row.items()}
        if values.get("A") == "236174" or values.get("B") == "16124":
            selected.append({
                "worksheet_row": re.sub(r"[^0-9]", "", next(iter(row))),
                **{name: values.get(column) for column, name in SELECTED_COLUMNS.items()},
            })

    join_path = ROOT / "brain_genotype_join.json"
    if join_path.stat().st_size > 1024**2:
        raise RuntimeError("retained non-outcome join exceeds bounded metadata read")
    join = json.loads(join_path.read_text(encoding="utf-8"))
    if (join.get("status") != "non_outcome_brain_genotype_join_only"
            or join.get("projection_class_decoded") is not False
            or join.get("morphology_or_targets_read") is not False):
        raise RuntimeError("named join must remain the permitted non-outcome metadata artifact")
    result = {
        "status": "provider_identity_observation_only",
        "job_id": job, "hostname": host,
        "source": str(source), "existing_join": str(join_path),
        "matching_provider_rows": selected,
        "existing_join_unmatched": [item for item in join["unmatched"] if item["brain_id"] == "236174"],
        "animal_236174_already_in_existing_join": any(item.get("animal_id") == "236174" for item in join["joined"]),
        "join_modified": False, "roles_or_eligibility_changed": False,
        "source_cell_workbook_or_morphology_or_outcomes_read": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
