#!/usr/bin/env python3
"""Steward feasibility join using a strict non-outcome column whitelist."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import xml.etree.ElementTree as ET
import zipfile

from inspect_public_brain_metadata import workbook_rows


def selected_cell_metadata(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    allowed = {"A": "cell_id", "B": "brain_id", "J": "soma_region", "L": "layer"}
    records, wanted = [], set()
    with zipfile.ZipFile(path) as z:
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("s:sheetData/s:row", ns):
            if row.attrib["r"] == "1":
                continue
            record = {}
            for cell in row.findall("s:c", ns):
                col = re.sub(r"[0-9]", "", cell.attrib["r"])
                if col not in allowed:
                    continue  # In particular, never decode Projection class (K).
                v = cell.find("s:v", ns)
                if v is None or v.text is None:
                    record[allowed[col]] = None
                elif cell.attrib.get("t") == "s":
                    record[allowed[col]] = (int(v.text),)
                    wanted.add(int(v.text))
                else:
                    record[allowed[col]] = v.text
            if record:
                records.append(record)
        strings = {}
        with z.open("xl/sharedStrings.xml") as f:
            index = 0
            for _, element in ET.iterparse(f, events=("end",)):
                if element.tag == "{" + ns["s"] + "}si":
                    if index in wanted:
                        strings[index] = "".join(element.itertext())
                    element.clear()
                    index += 1
                    if wanted <= strings.keys():
                        break
        for record in records:
            for key, value in record.items():
                if isinstance(value, tuple):
                    record[key] = strings[value[0]]
    return records


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    root = Path("outputs/runs/metadata_feasibility/attempt002")
    source = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition/seu_a1876_zenodo_13944322/archives/Full_morphometry.xlsx"
    cells = selected_cell_metadata(source)
    public = workbook_rows(root / "provider_brain_metadata.xlsx")["xl/worksheets/sheet1.xml"]
    brains = {}
    for row in public[1:]:
        values = {re.sub(r"[0-9]", "", key): value for key, value in row.items()}
        key = values.get("B")
        if not key:
            continue
        record = {"animal_id": values.get("A"), "image_id": key,
                  "technique": values.get("C"), "genotype": values.get("J"),
                  "genotype_provenance": values.get("K"), "target_description": values.get("L"),
                  "target_reference": values.get("M")}
        brains.setdefault(key, [])
        if record not in brains[key]:
            brains[key].append(record)
    counts = Counter(c.get("brain_id") for c in cells if c.get("cell_id"))
    joined, missing, ambiguous = [], [], []
    for brain, count in sorted(counts.items(), key=lambda x: str(x[0])):
        if brain not in brains:
            missing.append({"brain_id": brain, "cell_count": count})
            continue
        if len(brains[brain]) != 1:
            ambiguous.append({"brain_id": brain, "cell_count": count, "provider_records": brains[brain]})
            continue
        local = [c for c in cells if c.get("brain_id") == brain]
        joined.append({**brains[brain][0], "cell_count": count,
                       "source_layer_counts": [{"source_region": k[0], "layer": k[1], "cells": v}
                           for k, v in sorted(Counter((c.get("soma_region"), c.get("layer")) for c in local).items(), key=lambda x: str(x[0]))]})
    result = {"status": "non_outcome_brain_genotype_join_only", "job_id": os.environ["SLURM_JOB_ID"],
              "host": socket.gethostname(), "whitelisted_source_columns": ["A", "B", "J", "L"],
              "projection_class_decoded": False, "morphology_or_targets_read": False,
              "roles_assigned": False, "primary_eligibility_selected": False,
              "cell_count": sum(counts.values()), "brain_count": len(counts),
              "matched_brain_count": len(joined), "unmatched": missing,
              "ambiguous": ambiguous, "joined": joined}
    (root / "brain_genotype_join.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "joined"}, indent=2))
    print(json.dumps({"genotypes": dict(Counter(x["genotype"] for x in joined))}, indent=2))


if __name__ == "__main__":
    main()
