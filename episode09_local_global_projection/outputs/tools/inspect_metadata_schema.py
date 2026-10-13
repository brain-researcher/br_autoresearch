#!/usr/bin/env python3
"""Steward-only XLSX header census; never emit or interpret cell-row values."""
import hashlib
import json
import os
from pathlib import Path
import socket
import xml.etree.ElementTree as ET
import zipfile


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A verified Slurm compute step is required")
    root = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition/seu_a1876_zenodo_13944322"
    source = root / "archives/Full_morphometry.xlsx"
    out = Path("outputs/runs/metadata_feasibility/attempt001")
    out.mkdir(parents=True, exist_ok=False)
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(source) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = [{"name": x.attrib["name"], "sheet_id": x.attrib["sheetId"]}
                  for x in workbook.findall("s:sheets/s:sheet", ns)]
        headers = []
        for member in sorted(z.namelist()):
            if not (member.startswith("xl/worksheets/sheet") and member.endswith(".xml")):
                continue
            with z.open(member) as handle:
                for _, element in ET.iterparse(handle, events=("end",)):
                    if element.tag == "{" + ns["s"] + "}row":
                        cells = []
                        for cell in element.findall("s:c", ns):
                            v = cell.find("s:v", ns)
                            if cell.attrib.get("t") == "s" and v is not None:
                                cells.append((cell.attrib["r"], int(v.text)))
                            elif cell.attrib.get("t") == "inlineStr":
                                cells.append((cell.attrib["r"], "".join(cell.itertext())))
                        headers.append({"worksheet": member, "header_row": element.attrib["r"], "cells": cells})
                        break
        needed = {v for h in headers for _, v in h["cells"] if isinstance(v, int)}
        selected = {}
        with z.open("xl/sharedStrings.xml") as handle:
            index = 0
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag == "{" + ns["s"] + "}si":
                    if index in needed:
                        selected[index] = "".join(element.itertext())
                    element.clear()
                    index += 1
                    if needed <= selected.keys():
                        break
        for h in headers:
            h["cells"] = [{"column": col, "header": selected[v] if isinstance(v, int) else v} for col, v in h["cells"]]
    report = {"status": "metadata_schema_only", "source": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "sheets": sheets, "headers": headers,
              "cell_row_values_decoded": False, "roles_assigned": False,
              "outcome_independent_IT_definition_verified": False}
    (out / "metadata_schema.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
