#!/usr/bin/env python3
"""Write ledger v2: the shared exposure ledger plus EP09's frozen whole-animal roles.

Usage: add_ep09_roles_to_ledger.py LEDGER_CSV ROLE_MANIFEST_CSV OUT_CSV
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Slurm compute required")
    ledger_csv, roles_csv, out_csv = map(Path, sys.argv[1:4])
    roles = {r["fmost_id"]: r["ep09_role"] for r in csv.DictReader(open(roles_csv, newline=""))}
    rows = list(csv.DictReader(open(ledger_csv, newline="")))
    if out_csv.exists():
        raise SystemExit(f"refusing to overwrite {out_csv}")
    missing = set(roles) - {r["fmost_id"] for r in rows}
    if missing:
        raise SystemExit(f"role manifest brains absent from ledger: {sorted(missing)}")
    fields = list(rows[0].keys()) + ["ep09_role", "ep09_roles_frozen_from"]
    for r in rows:
        r["ep09_role"] = roles.get(r["fmost_id"], "none")
        r["ep09_roles_frozen_from"] = roles_csv.name if r["fmost_id"] in roles else ""
    with open(out_csv, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.chmod(out_csv, 0o440)
    digest = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    print(json.dumps({"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
                      "rows": len(rows), "ep09_audit": sum(r["ep09_role"] == "audit" for r in rows),
                      "ep09_development": sum(r["ep09_role"] == "development" for r in rows),
                      "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ledger_csv, roles_csv)},
                      "output_sha256": digest}, indent=1))


if __name__ == "__main__":
    main()
