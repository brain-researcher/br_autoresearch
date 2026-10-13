#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-interface
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
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
tools="$stage0/tools"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
development="$scratch_root/gao_mop_development_v1/swc"
source_assets="$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/allen_ccfv3_25um_2017_20260922/payload"
assets="$scratch_root/allen_ccfv3_25um_2017_ep10_steward_v3"
asset_partial="$scratch_root/.allen_ccfv3_25um_2017_ep10_steward_v3.${SLURM_JOB_ID}.part"
output="$scratch_root/v3_model_interface_preflight_v1"

for required in \
  "$python_bin" \
  "$tools/materialize_v3_model_interface.py" \
  "$tools/validate_v3_model_math_preflight.py" \
  "$stage0/projection_observation_contract.yaml" \
  "$stage0/projection_observation_contract_v2.yaml" \
  "$stage0/v3_model_interface_contract.yaml" \
  "$stage0/gao_mop_development_manifest.csv" \
  "$source_assets/annotation_25_ccf_2017.nrrd" \
  "$source_assets/structure_graph_1.json"; do
  test -f "$required"
done
test -d "$development"

if [[ -e "$output" ]]; then
  printf 'ERROR: refusing to overwrite interface preflight output: %s\n' "$output" >&2
  exit 3
fi

"$python_bin" --version
"$python_bin" "$tools/materialize_v3_model_interface.py" --self-test
"$python_bin" "$tools/validate_v3_model_math_preflight.py" \
  --self-test \
  --output "$scratch_root/v3_model_math_preflight_v1.json"

# Stage only the documented steward-acquisition payload into EP10 Scratch.
# No live sibling-episode output is accepted as an input.
if [[ ! -d "$assets" ]]; then
  if [[ -e "$asset_partial" ]]; then
    printf 'ERROR: refusing stale asset staging path: %s\n' "$asset_partial" >&2
    exit 4
  fi
  mkdir -p "$asset_partial"
  cp -- "$source_assets/annotation_25_ccf_2017.nrrd" "$asset_partial/"
  cp -- "$source_assets/structure_graph_1.json" "$asset_partial/"
  chmod 0440 "$asset_partial/annotation_25_ccf_2017.nrrd" "$asset_partial/structure_graph_1.json"
  mv -- "$asset_partial" "$assets"
fi
test -f "$assets/annotation_25_ccf_2017.nrrd"
test -f "$assets/structure_graph_1.json"

"$python_bin" "$tools/materialize_v3_model_interface.py" \
  --execute \
  --contract "$stage0/projection_observation_contract.yaml" \
  --revision-contract "$stage0/projection_observation_contract_v2.yaml" \
  --interface-contract "$stage0/v3_model_interface_contract.yaml" \
  --annotation "$assets/annotation_25_ccf_2017.nrrd" \
  --ontology "$assets/structure_graph_1.json" \
  --manifest "$stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development" \
  --output-root "$output"
