"""Render only two previously reported, authorized aggregate M1 gains."""
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

episode = Path("/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection")
source = episode / "outputs/experiment_log.md"
text = source.read_text()
m0_match = re.search(r"giving paired M1-minus-M0 gain\s+\x60([0-9.eE+-]+)\x60 bits per cell", text)
uniform_match = re.search(r"M1 beats the\s+fixed-K uniform reference by \x60([0-9.eE+-]+)\x60 bits per cell", text)
if m0_match is None or uniform_match is None:
    raise RuntimeError("Previously reported aggregate source values not found")
values = [float(m0_match.group(1)), float(uniform_match.group(1))]
stem = episode / "outputs/figures/v3_m1_reference_gains_exploratory_20261001"
with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.scatter(values, [1, 0], s=55, color=["#226B9D", "#777777"])
    ax.set_xscale("log")
    ax.set_xlim(0.001, 4)
    ax.set_ylim(-0.35, 1.45)
    ax.set_yticks([1, 0], ["M1 minus M0", "M1 minus fixed-K uniform"])
    ax.set_xlabel("Equal-group log-score gain (bits/cell; log scale)")
    ax.grid(axis="x", alpha=0.2)
    for value, y in zip(values, [1, 0]):
        ax.annotate(f"{value:.6g}", (value, y), xytext=(0, 10), textcoords="offset points", ha="center")
    fig.suptitle("Exploratory — existing M1 reference comparisons", y=0.99, fontsize=12)
    fig.text(0.02, 0.015, "194 development cells / 12 conservative groups. Different reference models.\nM1−M0 varies strongly by group; no uncertainty or M2 result is shown.", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(str(stem)+".png", dpi=180)
    fig.savefig(str(stem)+".svg")
    plt.close(fig)
metadata = {
    "classification": "exploratory",
    "source": str(source),
    "source_type": "existing authorized aggregate result in episode ledger",
    "new_data_access": False,
    "M1_minus_M0_equal_group_bits_per_cell": values[0],
    "M1_minus_uniform_equal_group_bits_per_cell": values[1],
    "cells": 194, "conservative_groups": 12,
    "limits": ["Different reference models", "No uncertainty intervals supplied",
               "M1 minus M0 is strongly heterogeneous by group", "No M2 or null result", "No biological promotion"]
}
Path(str(stem)+".json").write_text(json.dumps(metadata, indent=2)+"\n")
print(str(stem)+".png")

