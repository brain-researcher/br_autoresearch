#!/usr/bin/env python3
"""Join soma-only Gao MOp records to outcome-blind provider sample metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--somas", type=Path, required=True)
    parser.add_argument("--neurons", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.neurons.open("r", encoding="utf-8") as handle:
        neuron_payload = json.load(handle)
    safe_neurons: dict[str, dict[str, str]] = {}
    for provider_id, raw in neuron_payload["neuron_data"].items():
        acronym = str(raw.get("acronym", ""))
        if not acronym.startswith("MOp"):
            continue
        safe_neurons[str(provider_id)] = {
            "file": str(raw.get("file", "")),
        }
    if len(safe_neurons) != 837:
        raise RuntimeError(f"expected 837 safe MOp records, found {len(safe_neurons)}")

    with args.samples.open("r", encoding="utf-8") as handle:
        sample_rows = json.load(handle)
    samples: dict[str, dict[str, object]] = {}
    for raw in sample_rows:
        specimen_prefix = str(raw.get("fMOST_id", ""))
        if not specimen_prefix:
            continue
        if specimen_prefix in samples:
            raise RuntimeError(f"duplicate sample fMOST_id {specimen_prefix}")
        samples[specimen_prefix] = raw

    with args.somas.open("r", encoding="utf-8", newline="") as handle:
        soma_rows = list(csv.DictReader(handle))
    if len(soma_rows) != 837:
        raise RuntimeError(f"expected 837 soma rows, found {len(soma_rows)}")

    output_rows: list[dict[str, str]] = []
    for row in soma_rows:
        provider_id = row["provider_neuron_id"]
        safe = safe_neurons.get(provider_id)
        if safe is None or safe["file"] != row["file"]:
            raise RuntimeError(f"soma/neuron metadata mismatch for {provider_id}")
        specimen_prefix = row["filename_specimen_prefix"]
        sample = samples.get(specimen_prefix)
        if sample is None:
            raise RuntimeError(f"missing sample row for {specimen_prefix}")
        line = str(sample.get("transgenic_line", ""))
        if not line:
            raise RuntimeError(f"missing sample transgenic line for {specimen_prefix}")
        output_rows.append(
            {
                "provider_neuron_id": provider_id,
                "file": row["file"],
                "fMOST_brain_id": specimen_prefix,
                "provider_sample_id": str(sample.get("sample_id", "")),
                "source_acronym": row["source_acronym"],
                "strain_or_cre_line": line,
                "hemisphere": row["hemisphere"],
                "reconstruction_type": row["reconstruction_type"],
                "primary_axon_dendrite_distinguished_eligible": (
                    "yes"
                    if row["reconstruction_type"] == "Axon_and_dendrite"
                    else "no_axon_dendrite_not_distinguished"
                ),
                "soma_x_ccfv3_um": row["soma_x_ccfv3_um"],
                "soma_y_ccfv3_um": row["soma_y_ccfv3_um"],
                "soma_z_ccfv3_um": row["soma_z_ccfv3_um"],
                "soma_prefix_bytes_consumed": row["prefix_bytes_consumed"],
            }
        )

    brain_samples: dict[str, set[str]] = {}
    sample_brains: dict[str, set[str]] = {}
    for row in output_rows:
        brain_samples.setdefault(row["fMOST_brain_id"], set()).add(
            row["provider_sample_id"]
        )
        sample_brains.setdefault(row["provider_sample_id"], set()).add(
            row["fMOST_brain_id"]
        )
    if len(brain_samples) != 76 or any(len(value) != 1 for value in brain_samples.values()):
        raise RuntimeError("fMOST/sample mapping is not one-to-one from prefixes")
    if len(sample_brains) != 76 or any(len(value) != 1 for value in sample_brains.values()):
        raise RuntimeError("sample/fMOST mapping is not one-to-one from sample IDs")

    fieldnames = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(partial, args.output)

    line_groups: dict[str, set[str]] = {}
    for row in output_rows:
        line_groups.setdefault(row["strain_or_cre_line"], set()).add(
            row["fMOST_brain_id"]
        )
    print(
        json.dumps(
            {
                "cells": len(output_rows),
                "fMOST_sample_units": len(brain_samples),
                "primary_cells": sum(
                    row["primary_axon_dendrite_distinguished_eligible"] == "yes"
                    for row in output_rows
                ),
                "layer_counts": dict(
                    sorted(Counter(row["source_acronym"] for row in output_rows).items())
                ),
                "line_group_counts": {
                    line: len(groups) for line, groups in sorted(line_groups.items())
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
