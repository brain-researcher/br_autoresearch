#!/usr/bin/env python3
"""Write outputs/config/t9_freeze_manifest.json after the T11 development search selects a panel.

Refuses unless the T11-derived values and scientist decisions match the selected panel and positive
control, the known-positive control passed, and synthetic checks passed without skips.
The manifest covers code, runtime inputs and definitions, and records the acquisition manifest's
T9 archive size and expected SHA-256 for verification before opening T9.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

EPISODE_ROOT = Path(__file__).resolve().parents[2]
CODE = ["ep16_roles.py", "mat5_structural.py", "mat5_neural.py", "extract_development_features.py", "ep16_packets.py",
        "apply_source_fact_support.py", "apply_conditional_schedule_support.py", "describe_schedule_design.py",
        "describe_structural_support.py", "ep16_models.py", "ep16_panel.py", "ep16_branches.py", "ep16_control.py",
        "ep16_data.py", "evaluate_t9_sealed.py", "ep16_env.sh"]
RUNTIME_INPUTS = [
    Path("outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl"),
    Path("outputs/runs/structural_support/20261016-source_fact_support01/source_fact_support_report.json"),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquisition_archive():
    oak = os.environ.get("OAK")
    if not oak:
        raise SystemExit("OAK must identify the authorized acquisition location")
    source_root = Path(oak) / "data/br_autoresearch_data/braingate_long_term_array_performance/dryad-x0k6djj1h-v6"
    listing = source_root / "source_metadata/files.json"
    try:
        payload = json.loads(listing.read_text())
        embedded = payload.get("_embedded") if isinstance(payload, dict) else None
        entries = embedded.get("stash:files") if isinstance(embedded, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Expected _embedded['stash:files'] in the Dryad listing")
        matches = [entry for entry in entries
                   if isinstance(entry, dict) and entry.get("path") == "decoding_T9.tar.gz"]
        if len(matches) != 1:
            raise ValueError("Expected exactly one decoding_T9.tar.gz entry")
        entry = matches[0]
        if entry.get("digestType") != "sha-256":
            raise ValueError("The T9 acquisition digestType must be sha-256")
        size = entry.get("size")
        digest = entry.get("digest", entry.get("sha256"))
        if type(size) is not int or size <= 0:
            raise ValueError("The T9 acquisition size must be a positive integer")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("The T9 acquisition SHA-256 is missing or malformed")
        if "sha256" in entry and (not isinstance(entry["sha256"], str) or entry["sha256"].lower() != digest.lower()):
            raise ValueError("Conflicting T9 acquisition digest fields")
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Cannot freeze T9 archive identity: {error}") from error
    return {"path": str((source_root / "source/decoding_T9.tar.gz").resolve()),
            "size": size, "sha256": digest.lower(), "acquisition_manifest": str(listing.resolve())}


def main(argv=None):
    from evaluate_t9_sealed import frozen_choice_errors

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=Path("outputs/config/analysis_specification.json"))
    parser.add_argument("--selected-panel", type=Path, required=True)
    parser.add_argument("--positive-control", type=Path, required=True)
    parser.add_argument("--synthetic-checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/config/t9_freeze_manifest.json"))
    args = parser.parse_args(argv)
    spec = json.loads(args.specification.read_text())
    control = json.loads(args.positive_control.read_text())
    panel = json.loads(args.selected_panel.read_text())
    problems = frozen_choice_errors(spec, panel, control)
    if problems:
        raise SystemExit("Cannot freeze selected choices: " + "; ".join(problems))
    if control.get("pass") is not True:
        raise SystemExit("The T11 known-positive control did not pass")
    if panel.get("status") != "selected_by_development_search" or panel.get("rehearsal_only"):
        raise SystemExit("The selected panel is not a development-search selection")
    checks = args.synthetic_checks.read_text()
    if not re.search(r"^OK\b", checks, flags=re.M) or re.search(r"^FAILED", checks, flags=re.M) or "skipped" in checks:
        raise SystemExit("Synthetic checks did not all run and pass")
    archive = acquisition_archive()
    files = [args.specification, args.selected_panel, args.positive_control, args.synthetic_checks,
             EPISODE_ROOT / "outputs/config/analysis_definition.json",
             EPISODE_ROOT / "outputs/config/schedule_support.json",
             EPISODE_ROOT / "outputs/config/source_fact_rules.json"]
    files += [EPISODE_ROOT / "outputs/code" / c for c in CODE]
    files += [EPISODE_ROOT / path for path in RUNTIME_INPUTS]
    files += [Path(__file__)]
    files = [path.resolve() for path in files]
    if len(files) != len(set(files)):
        raise SystemExit("Freeze file list contains duplicate paths")
    manifest = {"episode": "EP16", "purpose": "freeze the single sealed T9 evaluation", "created_utc":
                datetime.now(timezone.utc).isoformat(), "specification": str(args.specification.resolve()),
                "selected_panel": str(args.selected_panel.resolve()), "positive_control": str(args.positive_control.resolve()),
                "synthetic_checks": str(args.synthetic_checks.resolve()), "t9_archive": archive,
                "files": [{"path": str(f), "sha256": sha256(f)} for f in files]}
    with open(args.output, "x") as handle:
        handle.write(json.dumps(manifest, indent=1) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
