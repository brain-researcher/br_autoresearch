#!/usr/bin/env python3
"""Download the public Gao whole-cortex SSp-tr SWCs and build a compact inventory.

The full public release is 13.6 GB.  EP11 needs only neurons whose provider
soma label is an SSp-tr layer, so this tool queries the official metadata and
downloads those SWCs individually.  Published projection-subtype labels are
deliberately omitted from every emitted metadata file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


SERVICE_ROOT = "https://mouse.digital-brain.cn/projectome/2/srv"
NEURON_METADATA_URL = f"{SERVICE_ROOT}/info/mouse/cortex/mouse.neuron.info.json"
SAMPLE_METADATA_URL = f"{SERVICE_ROOT}/info/mouse/cortex/sample.info.json"
SWC_ROOT = f"{SERVICE_ROOT}/info/mouse/cortex/neuron_download"
ALLOWED_SOURCE_LABELS = {
    "SSp-tr1",
    "SSp-tr2/3",
    "SSp-tr4",
    "SSp-tr5",
    "SSp-tr6a",
    "SSp-tr6b",
}
EXPECTED_TOTAL_SWC_BYTES = 332_357_190
SAFE_FILE = re.compile(r"^[A-Za-z0-9.-]+_[A-Za-z0-9.-]+\.swc$")


class AcquisitionError(RuntimeError):
    """Raised when provider metadata or a downloaded SWC fails validation."""


def get_json(url: str) -> Any:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.json()


def selected_metadata() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = get_json(NEURON_METADATA_URL)
    neuron_data = payload.get("neuron_data")
    if not isinstance(neuron_data, dict):
        raise AcquisitionError("official neuron metadata lacks neuron_data")

    selected: list[dict[str, Any]] = []
    for provider_id, raw in neuron_data.items():
        if raw.get("acronym") not in ALLOWED_SOURCE_LABELS:
            continue
        filename = str(raw.get("file", ""))
        if not SAFE_FILE.fullmatch(filename):
            raise AcquisitionError(f"unsafe or unexpected SWC filename: {filename!r}")
        brain_id = filename.split("_", 1)[0]
        type_array = raw.get("type_array")
        if not isinstance(type_array, list) or len(type_array) != 1:
            raise AcquisitionError(f"unexpected type_array for {filename}")
        selected.append(
            {
                "provider_neuron_id": str(provider_id),
                "file": filename,
                "fMOST_brain_id": brain_id,
                "source_acronym": str(raw.get("acronym")),
                "source_name": str(raw.get("name", "")),
                "strain_or_cre_line": str(type_array[0]),
                "hemisphere": str(raw.get("hemisphere", "")),
                "reconstruction_type": str(raw.get("reconstruction_type", "")),
            }
        )

    selected.sort(key=lambda row: (row["fMOST_brain_id"], row["file"]))
    if len(selected) != 214:
        raise AcquisitionError(f"expected 214 SSp-tr neurons, found {len(selected)}")
    if len({row["file"] for row in selected}) != len(selected):
        raise AcquisitionError("duplicate SSp-tr SWC filename")
    if len({row["fMOST_brain_id"] for row in selected}) != 36:
        raise AcquisitionError("expected 36 distinct SSp-tr fMOST brain IDs")

    sample_rows = get_json(SAMPLE_METADATA_URL)
    if not isinstance(sample_rows, list):
        raise AcquisitionError("official sample metadata is not a list")
    samples: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        brain_id = str(row.get("fMOST_id", ""))
        if not brain_id:
            continue
        if brain_id in samples:
            raise AcquisitionError(f"duplicate fMOST sample metadata: {brain_id}")
        samples[brain_id] = row
    missing = sorted({row["fMOST_brain_id"] for row in selected} - set(samples))
    if missing:
        raise AcquisitionError(f"SSp-tr brain IDs absent from sample metadata: {missing}")
    return selected, samples


def download_one(row: dict[str, Any], swc_root: Path) -> tuple[str, str]:
    brain_id = row["fMOST_brain_id"]
    filename = row["file"]
    destination = swc_root / brain_id / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return filename, "existing"

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{SWC_ROOT}/{filename}"
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with requests.get(url, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise AcquisitionError(f"empty SWC response for {filename}")
            os.replace(temporary, destination)
            return filename, "downloaded"
        except Exception as error:  # retry provider/network failures
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(attempt * 2)
    raise AcquisitionError(f"failed to download {filename}: {last_error}")


def inspect_swc(path: Path) -> dict[str, Any]:
    node_types: Counter[int] = Counter()
    soma: tuple[float, float, float] | None = None
    node_count = 0
    node_ids: set[int] = set()
    parents: list[int] = []
    crc32 = 0
    with path.open("rb") as binary_handle:
        for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
            crc32 = zlib.crc32(chunk, crc32)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 7:
                raise AcquisitionError(f"malformed SWC row {path}:{line_number}")
            try:
                node_id = int(fields[0])
                node_type = int(float(fields[1]))
                x, y, z = (float(fields[index]) for index in (2, 3, 4))
                parent_id = int(fields[6])
            except ValueError as error:
                raise AcquisitionError(
                    f"non-numeric SWC row {path}:{line_number}"
                ) from error
            if node_id in node_ids:
                raise AcquisitionError(f"duplicate SWC node ID {path}:{node_id}")
            node_ids.add(node_id)
            parents.append(parent_id)
            node_types[node_type] += 1
            node_count += 1
            if soma is None and node_type == 1:
                soma = (x, y, z)
    if node_count == 0 or soma is None:
        raise AcquisitionError(f"SWC lacks nodes or soma: {path}")
    # This release uses both 0 and standard SWC 2 for axon, depending on the
    # reconstruction pipeline.  Type 3 is the distinguished dendrite tree.
    # Provider-labelled Axon_only files explicitly do not distinguish the two
    # trees and are downloaded for sensitivity/QC but are not primary-eligible.
    axon_nodes = node_types[0] + node_types[2]
    if axon_nodes == 0:
        raise AcquisitionError(f"SWC lacks a provider axon tree: {path}")
    if parents.count(-1) != 1:
        raise AcquisitionError(f"SWC must have exactly one root: {path}")
    missing_parents = sorted({parent for parent in parents if parent != -1} - node_ids)
    if missing_parents:
        raise AcquisitionError(
            f"SWC has missing parent IDs in {path}: {missing_parents[:5]}"
        )
    return {
        "file_size_bytes": path.stat().st_size,
        "node_count": node_count,
        "soma_x_ccfv3_um": soma[0],
        "soma_y_ccfv3_um": soma[1],
        "soma_z_ccfv3_um": soma[2],
        "soma_nodes": node_types[1],
        "provider_axon_nodes_type_0_or_2": axon_nodes,
        "provider_dendrite_nodes_type_3": node_types[3],
        "other_marker_nodes_type_10_11_12": (
            node_types[10] + node_types[11] + node_types[12]
        ),
        "crc32_hex": f"{crc32 & 0xffffffff:08x}",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise AcquisitionError("workers must be between 1 and 32")
    args.output_root.mkdir(parents=True, exist_ok=True)
    swc_root = args.output_root / "swc"
    selected, samples = selected_metadata()

    download_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, swc_root): row["file"]
            for row in selected
        }
        for future in as_completed(futures):
            _, status = future.result()
            download_counts[status] += 1

    inventory: list[dict[str, Any]] = []
    for row in selected:
        sample = samples[row["fMOST_brain_id"]]
        if str(sample.get("transgenic_line", "")) != row["strain_or_cre_line"]:
            raise AcquisitionError(
                f"line mismatch for fMOST brain {row['fMOST_brain_id']}"
            )
        inspected = inspect_swc(
            swc_root / row["fMOST_brain_id"] / row["file"]
        )
        inventory.append(
            {
                **row,
                "provider_sample_id": str(sample.get("sample_id", "")),
                "sample_gender_code": str(sample.get("gender", "")),
                "sample_age_days": str(sample.get("age", "")),
                "sample_lab_name": str(sample.get("lab_name", "")),
                "sample_author": str(sample.get("author", "")),
                "sample_tracing_cell_number": str(
                    sample.get("tracing_cell_number", "")
                ),
                "sample_injection_region": str(sample.get("injection_region", "")),
                "sample_injection_structure": str(
                    sample.get("injection_structure", "")
                ),
                "primary_axon_dendrite_distinguished_eligible": (
                    "yes"
                    if row["reconstruction_type"] == "Axon_and_dendrite"
                    else "no_axon_dendrite_not_distinguished"
                ),
                **inspected,
            }
        )

    fields = list(inventory[0])
    args.inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)

    filtered_samples = {
        brain_id: {
            "fMOST_id": brain_id,
            "gender": sample.get("gender"),
            "age": sample.get("age"),
            "lab_name": sample.get("lab_name"),
            "author": sample.get("author"),
            "transgenic_line": sample.get("transgenic_line"),
            "tracing_cell_number": sample.get("tracing_cell_number"),
            "injection_region": sample.get("injection_region"),
            "injection_structure": sample.get("injection_structure"),
        }
        for brain_id, sample in samples.items()
        if brain_id in {row["fMOST_brain_id"] for row in selected}
    }
    write_json(args.output_root / "selected_neuron_metadata.json", selected)
    write_json(args.output_root / "selected_sample_metadata.json", filtered_samples)

    layer_counts = Counter(row["source_acronym"] for row in inventory)
    line_cells = Counter(row["strain_or_cre_line"] for row in inventory)
    line_brains = {
        line: len(
            {
                row["fMOST_brain_id"]
                for row in inventory
                if row["strain_or_cre_line"] == line
            }
        )
        for line in sorted(line_cells)
    }
    summary = {
        "provider": "Gao et al. whole-cortex projectome",
        "provider_neuron_metadata_url": NEURON_METADATA_URL,
        "provider_sample_metadata_url": SAMPLE_METADATA_URL,
        "provider_swc_root": SWC_ROOT,
        "selection_rule": "provider source_acronym in exact SSp-tr layer labels",
        "cells": len(inventory),
        "fMOST_brain_samples": len({row["fMOST_brain_id"] for row in inventory}),
        "layer_cell_counts": dict(sorted(layer_counts.items())),
        "line_cell_counts": dict(sorted(line_cells.items())),
        "line_brain_counts": line_brains,
        "reconstruction_type_counts": dict(
            sorted(Counter(row["reconstruction_type"] for row in inventory).items())
        ),
        "total_downloaded_bytes": sum(
            int(row["file_size_bytes"]) for row in inventory
        ),
        "download_actions": dict(sorted(download_counts.items())),
        "all_files_have_soma_and_axon_nodes": all(
            int(row["soma_nodes"]) >= 1
            and int(row["provider_axon_nodes_type_0_or_2"]) >= 1
            for row in inventory
        ),
        "primary_axon_dendrite_distinguished_eligible_cells": sum(
            row["primary_axon_dendrite_distinguished_eligible"] == "yes"
            for row in inventory
        ),
    }
    if summary["total_downloaded_bytes"] != EXPECTED_TOTAL_SWC_BYTES:
        raise AcquisitionError(
            "downloaded SWC byte total differs from the official Zenodo ZIP "
            f"directory: {summary['total_downloaded_bytes']} != "
            f"{EXPECTED_TOTAL_SWC_BYTES}"
        )
    write_json(args.summary_json, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
