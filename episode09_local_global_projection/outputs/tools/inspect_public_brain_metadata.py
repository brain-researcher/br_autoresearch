#!/usr/bin/env python3
"""Inspect the newly acquired public brain/genotype supplement, not neuron outcomes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import xml.etree.ElementTree as ET
import zipfile


def workbook_rows(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        result = {}
        for member in sorted(z.namelist()):
            if not (member.startswith("xl/worksheets/sheet") and member.endswith(".xml")):
                continue
            rows = []
            for row in ET.fromstring(z.read(member)).findall("s:sheetData/s:row", ns):
                record = {}
                for cell in row.findall("s:c", ns):
                    v = cell.find("s:v", ns)
                    if cell.attrib.get("t") == "s" and v is not None:
                        value = shared[int(v.text)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(cell.itertext())
                    else:
                        value = None if v is None else v.text
                    record[cell.attrib["r"]] = value
                rows.append(record)
            result[member] = rows
        return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    root = Path("outputs/runs/metadata_feasibility/attempt002")
    source = root / "provider_brain_metadata.xlsx"
    rows = workbook_rows(source)
    result = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "source_url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-54745-6/MediaObjects/41467_2024_54745_MOESM3_ESM.xlsx",
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "sheets": {k: {"row_count": len(v), "first_five_rows": v[:5]} for k, v in rows.items()},
              "SEU_cell_rows_read": False}
    (root / "public_metadata_schema.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
