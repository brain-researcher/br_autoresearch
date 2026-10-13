#!/usr/bin/env bash
#SBATCH --job-name=ep10-v2-targets
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
script="$stage0/tools/construct_mop_terminal_targets.py"
assets="$SCRATCH/br_autoresearch/episode11_projection_types_vs_gradients/allen_outcome_assets"
development="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1"
qualification="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_structural_qc_v2/aggregate_summary.json"
output="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_targets_v2"

"$python_bin" --version
"$python_bin" - "$qualification" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    summary = json.load(handle)
required = {
    "status": "complete",
    "stage": "provider_native_v2_structural_qualification",
    "swc_interpretation": "provider_native_topology_v2",
    "input_files": 200,
    "input_groups": 12,
    "minimum_overall_eligible_cells": 139,
    "structural_QC_gate_passed": True,
    "target_construction_authorized": True,
    "target_vocabulary_loaded": False,
    "projection_outcomes_computed": False,
    "projection_outcomes_emitted": False,
    "final_or_unassigned_files_opened": False,
}
mismatches = {
    key: {"expected": expected, "observed": summary.get(key)}
    for key, expected in required.items()
    if summary.get(key) != expected
}
if mismatches:
    raise SystemExit(f"v2 structural qualification did not authorize targets: {mismatches}")
group_pass = summary.get("group_gate_pass", {})
positive_groups = summary.get("positive_capacity_groups", [])
if len(positive_groups) != 11 or not all(group_pass.get(group) is True for group in positive_groups):
    raise SystemExit("v2 positive-capacity group gate did not pass")
print("v2_structural_gate=pass")
PY

test -f "$script"
test -f "$stage0/projection_observation_contract.yaml"
test -f "$stage0/projection_observation_contract_v2.yaml"
test -f "$stage0/allen_target_vocabulary.csv"
test -f "$assets/annotation_25_ccf_2017.nrrd"
test -f "$assets/structure_graph_1.json"
test -f "$development/download_log.csv"

"$python_bin" "$script" \
  --execute \
  --contract "$stage0/projection_observation_contract.yaml" \
  --revision-contract "$stage0/projection_observation_contract_v2.yaml" \
  --vocabulary "$stage0/allen_target_vocabulary.csv" \
  --annotation "$assets/annotation_25_ccf_2017.nrrd" \
  --ontology "$assets/structure_graph_1.json" \
  --manifest "$stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development/swc" \
  --output-root "$output"
