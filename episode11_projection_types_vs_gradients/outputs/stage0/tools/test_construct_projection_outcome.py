#!/usr/bin/env python3
"""Outcome-blind regression test for construct_projection_outcome.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from construct_projection_outcome import (
    ConstructionError,
    _load_manifest,
    run_self_test,
)


def test_manifest_hemisphere_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        valid = directory / "valid.csv"
        valid.write_text(
            "cell_id,swc_path,hemisphere\ncell_1,relative/cell_1.swc,Left\n",
            encoding="utf-8",
        )
        entries = _load_manifest(valid, None, None, None)
        if len(entries) != 1:
            raise AssertionError("valid manifest did not yield exactly one entry")
        entry = entries[0]
        if entry.provider_hemisphere != "left":
            raise AssertionError("provider hemisphere was not normalized")
        if entry.swc_path != directory / "relative/cell_1.swc":
            raise AssertionError("relative SWC path was not resolved from manifest")

        missing = directory / "missing.csv"
        missing.write_text(
            "cell_id,swc_path\ncell_1,cell_1.swc\n", encoding="utf-8"
        )
        try:
            _load_manifest(missing, None, None, None)
        except ConstructionError:
            pass
        else:
            raise AssertionError("manifest without provider hemisphere was accepted")

        conflicting = directory / "conflicting.csv"
        conflicting.write_text(
            "cell_id,swc_path,provider_hemisphere,hemisphere\n"
            "cell_1,cell_1.swc,Left,Right\n",
            encoding="utf-8",
        )
        try:
            _load_manifest(conflicting, None, None, None)
        except ConstructionError:
            pass
        else:
            raise AssertionError("conflicting provider hemisphere columns were accepted")

        truncated = directory / "truncated.csv"
        truncated.write_text(
            "cell_id,swc_path,hemisphere\ncell_1,cell_1.swc\n",
            encoding="utf-8",
        )
        try:
            _load_manifest(truncated, None, None, None)
        except ConstructionError:
            pass
        else:
            raise AssertionError("manifest row with missing hemisphere was accepted")


def main() -> int:
    result = run_self_test()
    if result["status"] != "pass" or result["uses_real_atlas_or_swc"] is not False:
        raise AssertionError("projection constructor synthetic fixture did not pass")
    test_manifest_hemisphere_contract()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
