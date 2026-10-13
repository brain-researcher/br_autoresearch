#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m1-geometry
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
engine="$stage0/tools/materialize_v3_m1_geometry.py"
contract="$stage0/v3_m1_geometry_contract.yaml"
observation_contract="$stage0/projection_observation_contract.yaml"
vocabulary="$stage0/allen_target_vocabulary.csv"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
atlas_root="$scratch_root/allen_ccfv3_25um_2017_ep10_steward_v3"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
output_root="$scratch_root/v3_m1_geometry_v1"

test -x "$python_bin"

for required in \
  "$engine" \
  "$contract" \
  "$observation_contract" \
  "$vocabulary" \
  "$atlas_root/annotation_25_ccf_2017.nrrd" \
  "$atlas_root/structure_graph_1.json" \
  "$interface_root/v3_model_interface.csv" \
  "$interface_root/v3_model_interface_summary.json"; do
  test -f "$required"
  test ! -L "$required"
done

if [[ -e "$output_root" ]]; then
  printf 'ERROR: refusing existing M1 geometry output: %s\n' "$output_root" >&2
  exit 3
fi

"$python_bin" --version
"$python_bin" "$engine" --self-test
"$python_bin" "$engine" \
  --execute \
  --contract "$contract" \
  --observation-contract "$observation_contract" \
  --annotation "$atlas_root/annotation_25_ccf_2017.nrrd" \
  --ontology "$atlas_root/structure_graph_1.json" \
  --vocabulary "$vocabulary" \
  --model-interface "$interface_root/v3_model_interface.csv" \
  --output-root "$output_root"
