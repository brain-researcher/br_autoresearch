#!/usr/bin/env bash
#SBATCH --job-name=ep10-v2-struct-qc
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
output="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_structural_qc_v2"

"$python_bin" --version
"$python_bin" - <<'PY'
from importlib.metadata import version

for distribution in ("numpy", "PyYAML", "pynrrd", "typing_extensions"):
    print(f"{distribution}={version(distribution)}")
PY

test -f "$script"
test -f "$stage0/projection_observation_contract.yaml"
test -f "$stage0/projection_observation_contract_v2.yaml"
test -f "$assets/annotation_25_ccf_2017.nrrd"
test -f "$assets/structure_graph_1.json"
test -f "$development/download_log.csv"

"$python_bin" "$script" \
  --execute \
  --structural-qc-only \
  --contract "$stage0/projection_observation_contract.yaml" \
  --revision-contract "$stage0/projection_observation_contract_v2.yaml" \
  --annotation "$assets/annotation_25_ccf_2017.nrrd" \
  --ontology "$assets/structure_graph_1.json" \
  --manifest "$stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development/swc" \
  --output-root "$output"
