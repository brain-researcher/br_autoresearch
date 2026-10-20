#!/usr/bin/env python3
"""Catalog-only survey: BIL cell-morphology datasets from IT driver lines.

Uses the BIL metadata API (no data download). Counts unique animals and
reconstruction datasets per genotype, and flags animals already in SEU-A1876.
Usage: survey_bil_it_reconstructions.py JOIN_JSON OUT_JSON
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.brainimagelibrary.org"
LINES = ["Cux2", "Plxnd1", "Tlx3", "Rorb", "Sepw1", "Selenow", "Otof", "Slc30a3",
         "Nr5a1", "Scnn1a", "Rasgrf2"]


def get(url, data=None, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:  # the API answers 404 for an empty result
                return {}
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def norm(x):
    s = str(x or "").strip()
    return s[:-2] if s.endswith(".0") else s


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    join_json, out = Path(sys.argv[1]), Path(sys.argv[2])
    join = json.loads(join_json.read_text())["joined"]
    seu_animals = {x["animal_id"] for x in join} | {"236174"}
    seu_brains = {x["image_id"] for x in join} | {"236174", "16124"}
    ids_by_line = {}
    for line in LINES:
        d = get(f"{API}/query/specimen?genotype=" + urllib.parse.quote(line))
        ids_by_line[line] = [i for i in d.get("bildids", []) if i.startswith("ace-")]
    all_ids = sorted({i for v in ids_by_line.values() for i in v})
    records = {}
    for k in range(0, len(all_ids), 40):
        batch = all_ids[k:k + 40]
        d = get(f"{API}/retrieve", data=json.dumps({"bildids": batch}).encode())
        for r in d.get("retjson", []):
            ds = r["Dataset"][0] if isinstance(r.get("Dataset"), list) and r["Dataset"] else (r.get("Dataset") or {})
            bid = (r.get("Assets") or [{}])[0].get("bildid") or ds.get("bildid")
            records[bid] = {
                "title": ds.get("title"), "modality": ds.get("generalmodality"), "technique": ds.get("technique"),
                "n_files": ds.get("number_of_files"), "bildirectory": ds.get("bildirectory"),
                "project": (r.get("Submission") or {}).get("project"),
                "bildate": (r.get("Submission") or {}).get("bildate"),
                "specimens": [{"animal": norm(s.get("localid")), "brain": norm(s.get("organlocalid") or s.get("samplelocalid")),
                               "genotype": s.get("genotype"), "locations": s.get("locations")}
                              for s in (r.get("Specimen") or [])]}
    per_line = {}
    for line, ids in ids_by_line.items():
        recon = [records[i] for i in ids if i in records and (records[i]["modality"] or "").lower() == "cell morphology"]
        animals = defaultdict(lambda: {"datasets": 0, "single_cell_datasets": 0, "per_brain_files": 0,
                                       "locations": [], "genotypes": set(), "in_seu_a1876": False})
        for rec in recon:
            single = (rec["title"] or "").startswith("Single whole neuron")
            for s in rec["specimens"]:
                if line.lower() not in (s["genotype"] or "").lower():
                    continue
                a = animals[s["animal"] or s["brain"]]
                a["datasets"] += 1
                a["single_cell_datasets"] += int(single)
                if not single:
                    try:
                        a["per_brain_files"] += int(float(rec["n_files"] or 0))
                    except ValueError:
                        pass
                if single and s["locations"]:
                    a["locations"].append(s["locations"])
                a["genotypes"].add(s["genotype"])
                a["in_seu_a1876"] |= (s["animal"] in seu_animals) or (s["brain"] in seu_brains)
        per_line[line] = {
            "datasets_matching_genotype": len(ids), "cell_morphology_datasets": len(recon),
            "animals": {k: {**v, "genotypes": sorted(g for g in v["genotypes"] if g)} for k, v in animals.items()},
            "animals_total": len(animals),
            "animals_not_in_seu_a1876": sorted(k for k, v in animals.items() if not v["in_seu_a1876"])}
    new_animals = {a for v in per_line.values() for a in v["animals_not_in_seu_a1876"]}
    new_records = {bid: {k: rec[k] for k in ("title", "modality", "technique", "n_files", "project", "bildate")}
                   | {"animals": sorted({s["animal"] for s in rec["specimens"]})}
                   for bid, rec in records.items()
                   if (rec["modality"] or "").lower() == "cell morphology"
                   and any(s["animal"] in new_animals for s in rec["specimens"])}
    report = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "source": "BIL metadata API (catalog only; no data files)", "lines": LINES,
              "per_line": per_line, "new_animal_records": new_records}
    out.write_text(json.dumps(report, indent=1, default=list) + "\n")
    for line, v in per_line.items():
        print(line, "| genotype datasets", v["datasets_matching_genotype"], "| morphology", v["cell_morphology_datasets"],
              "| animals", v["animals_total"], "| not in SEU-A1876", len(v["animals_not_in_seu_a1876"]))


if __name__ == "__main__":
    main()
