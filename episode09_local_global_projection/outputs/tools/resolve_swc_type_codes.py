#!/usr/bin/env python3
"""Pool SWC topology for Cux2, Wfs1 and Scnn1a brains."""

import argparse
import atexit
import json
import multiprocessing
import os
import re
import socket
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ZIP_HANDLE = None
CHUNK_SIZE = 64
FOCUS_TYPES = {0, 10, 11, 12}

# Capture only ID, type and parent. The four intervening fields are opaque:
# coordinates and radius are never converted or retained in node records.
TOPOLOGY_ROW = re.compile(
    rb"\s*([+-]?\d+)\s+([+-]?\d+)(?:\s+\S+){4}\s+([+-]?\d+)\s*"
)


def normalize_id(value):
    if value is None:
        raise ValueError("A brain identifier is null.")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        raise ValueError("A brain identifier is empty.")
    return text


def file_fields_only(obj):
    # Discard other fields when decoding each neuron record.
    if isinstance(obj.get("file"), str):
        return {"file": obj["file"]}
    return obj


def initialize_worker(zip_path):
    global ZIP_HANDLE
    ZIP_HANDLE = zipfile.ZipFile(zip_path, "r")
    atexit.register(ZIP_HANDLE.close)


def read_topology(member, issues):
    nodes = {}
    with ZIP_HANDLE.open(member, "r") as handle:
        for raw in handle:
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            parts = raw.split()
            if len(parts) < 7:
                issues["short_rows"] += 1
                continue
            try:  # only id, type and parent are converted; coordinates stay unparsed
                node_id, node_type, parent_id = (int(float(parts[0])), int(float(parts[1])), int(float(parts[6])))
            except ValueError:
                issues["unparsable_rows"] += 1
                continue
            if node_id in nodes:
                issues["duplicate_node_ids"] += 1
                continue
            nodes[node_id] = (node_type, parent_id)
    return nodes


def process_chunk(members):
    edges, soma_children, files_with_type, issues = Counter(), Counter(), Counter(), Counter()
    processed = 0
    for member in members:
        nodes = read_topology(member, issues)
        files_with_type.update({node_type for node_type, _ in nodes.values()})
        for node_type, parent_id in nodes.values():
            if parent_id == -1:
                parent_type = "root"
            elif parent_id not in nodes:
                parent_type = "missing"
            else:
                parent_type = nodes[parent_id][0]
            edges[(node_type, parent_type)] += 1
            if parent_type == 1:
                soma_children[node_type] += 1
        processed += 1

    # Only chunk-pooled totals cross the process boundary.
    return edges, soma_children, files_with_type, processed, issues


def main():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.strip():
        raise SystemExit("Refusing to run: SLURM_JOB_ID must be set.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip")
    parser.add_argument("sample_json")
    parser.add_argument("neuron_json")
    parser.add_argument("out_json")
    parser.add_argument("nproc", nargs="?", type=int)
    args = parser.parse_args()

    try:
        cpu_limit = int(os.environ.get("SLURM_CPUS_PER_TASK") or "1")
    except ValueError:
        parser.error("SLURM_CPUS_PER_TASK must be a positive integer.")
    if hasattr(os, "sched_getaffinity"):
        cpu_limit = min(cpu_limit, len(os.sched_getaffinity(0)))
    nproc = cpu_limit if args.nproc is None else args.nproc
    if not 1 <= nproc <= cpu_limit:
        parser.error(f"NPROC must be between 1 and {cpu_limit}.")

    with Path(args.sample_json).open(encoding="utf-8-sig") as handle:
        samples = json.load(handle)
    lines = {}
    for sample in samples:
        brain = normalize_id(sample["fMOST_id"])
        line = str(sample["transgenic_line"]).casefold()
        if brain in lines and lines[brain] != line:
            raise ValueError("Conflicting transgenic lines for a brain.")
        lines[brain] = line
    eligible = {
        brain for brain, line in lines.items()
        if any(marker in line for marker in ("cux2", "wfs1", "scnn1a"))
    }
    del samples

    with Path(args.neuron_json).open(encoding="utf-8-sig") as handle:
        neuron_data = json.load(
            handle, object_hook=file_fields_only
        )["neuron_data"]

    requested = set()
    for record in neuron_data.values():
        filename = record["file"]  # Do not access any other record field.
        brain, separator, cell_file = filename.partition("_")
        if (
            not separator or "/" in filename or "\\" in filename
            or not cell_file.endswith(".swc") or len(cell_file) <= 4
        ):
            raise ValueError("Invalid NEURON_JSON file name.")
        brain = normalize_id(brain)
        if brain in eligible:
            requested.add((brain, cell_file))
    del neuron_data

    # Resolve normalized brain IDs to actual archive member names.
    zip_path = str(Path(args.zip).resolve())
    index = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 3 or parts[0] != "data_v3":
                continue
            key = (normalize_id(parts[1]), parts[2])
            if key not in requested:
                continue
            if key in index:
                raise ValueError("Duplicate ZIP members for a requested cell.")
            index[key] = info.filename

    if requested.difference(index):
        raise ValueError("ZIP is missing at least one eligible requested SWC.")

    members = [index[key] for key in sorted(requested)]
    chunks = [
        members[start:start + CHUNK_SIZE]
        for start in range(0, len(members), CHUNK_SIZE)
    ]
    edges, soma_children, files_with_type, issues = Counter(), Counter(), Counter(), Counter()
    files_processed = 0

    if chunks:
        with ProcessPoolExecutor(
            max_workers=min(nproc, len(chunks)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(zip_path,),
        ) as executor:
            for chunk_edges, chunk_soma, chunk_files, count, chunk_issues in executor.map(
                process_chunk, chunks, chunksize=1
            ):
                issues.update(chunk_issues)
                edges.update(chunk_edges)
                soma_children.update(chunk_soma)
                files_with_type.update(chunk_files)
                files_processed += count

    types = sorted(set(files_with_type) | FOCUS_TYPES)
    parent_types = types + ["root", "missing"]
    result = {
        "SLURM_JOB_ID": job_id,
        "hostname": socket.gethostname(),
        "files_processed": files_processed,
        "pooled_row_issues": dict(issues),
        "total_nodes": sum(edges.values()),
        "node_counts_child_type_by_parent_type": {
            str(child): {
                str(parent): edges[(child, parent)]
                for parent in parent_types
            }
            for child in types
        },
        "nodes_with_soma_parent_by_type": {
            str(node_type): soma_children[node_type] for node_type in types
        },
        "files_containing_type": {
            str(node_type): files_with_type[node_type] for node_type in types
        },
    }

    out_path = Path(args.out_json)
    if out_path.resolve() in {
        Path(args.zip).resolve(),
        Path(args.sample_json).resolve(),
        Path(args.neuron_json).resolve(),
    }:
        raise ValueError("OUT_JSON must differ from every input file.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
