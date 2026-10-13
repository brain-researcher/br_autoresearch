#!/usr/bin/env python3
"""EP21 compatibility snapshot export, isolated from its ASTRA program."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from brain_researcher.autoresearch.astra_export import ASTRA_EXPORT_VERSION
from brain_researcher.autoresearch.astra_materialization import (
    ASTRA_PROJECTION_FILENAME,
    AstraMaterializationError,
    write_astra_projection,
)
from brain_researcher.autoresearch.astra_workspace import build_episode_astra_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "terminal"))
    parser.add_argument("--snapshot-id", help="Named revision; preserve earlier snapshots after a contract amendment.")
    parser.add_argument("--output-ref", action="append", default=[])
    parser.add_argument("--finding")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    if args.snapshot_id and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.snapshot_id):
        parser.error("snapshot-id must contain only lowercase letters, digits, underscores or hyphens")
    # Derive the one authorized workspace; accept no caller-selected workspace
    # or destination. The full program is never a compatibility write target.
    workspace = Path(__file__).resolve().parents[2]
    directory = workspace / "outputs/astra/compatibility" / f"v{ASTRA_EXPORT_VERSION}" / args.stage
    if args.snapshot_id:
        directory /= args.snapshot_id
    path = directory / ASTRA_PROJECTION_FILENAME
    try:
        document = build_episode_astra_document(
            workspace=workspace,
            stage=args.stage,
            output_refs=tuple(args.output_ref),
            finding=args.finding,
            created_at=args.created_at,
        )
        created, replay, warnings = write_astra_projection(
            path=path,
            document=document,
            replace=args.stage == "terminal",
            trusted_root=workspace,
        )
    except AstraMaterializationError as exc:
        print(json.dumps({"ok": False, "error": exc.code,
                          "message": str(exc)}))
        return 2
    print(json.dumps({
        "ok": True,
        "stage": args.stage,
        "astra_version": ASTRA_EXPORT_VERSION,
        "projection_path": str(path),
        "created": created,
        "idempotent_replay": replay,
        "validation_warnings": list(warnings),
        "record_role": "legacy_compatibility_snapshot_only",
        "scientific_authority": False,
        "experiment_program": str(workspace / "outputs/astra" /
                                  f"v{ASTRA_EXPORT_VERSION}" / "astra.yaml"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
