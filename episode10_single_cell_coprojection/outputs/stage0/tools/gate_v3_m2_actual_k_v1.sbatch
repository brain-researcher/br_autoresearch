#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-actual-k
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

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
m1_engine="$stage0/tools/fit_v3_m1.py"
m2_engine="$stage0/tools/fit_v3_m2.py"
m1_projection_gate="$stage0/tools/gate_v3_m1_actual_k.py"
gate="$stage0/tools/gate_v3_m2_actual_k.py"
m1_projection_test="$stage0/tests/test_gate_v3_m1_actual_k.py"
test_file="$stage0/tests/test_gate_v3_m2_actual_k.py"
contract="$stage0/v3_m2_search_contract.yaml"
addendum="$stage0/v3_m2_numerical_falsifier_addendum_v1.yaml"
inventory="$stage0/v3_m2_frozen_cell_inventory.tsv"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
core_root="$scratch_root/v3_m2_core_contract_qualification_job_45832824"
core_report="$core_root/v3_m2_core_contract_qualification.json"
output_root="$scratch_root/v3_m2_actual_k_gate_v1_job_${SLURM_JOB_ID}"

test -x "$python_bin"
test -d "$scratch_root"
test ! -e "$output_root"

for required in \
  "$m1_engine" \
  "$m2_engine" \
  "$m1_projection_gate" \
  "$gate" \
  "$m1_projection_test" \
  "$test_file" \
  "$contract" \
  "$addendum" \
  "$inventory" \
  "$core_report" \
  "$interface_root/v3_model_interface.csv" \
  "$interface_root/v3_model_interface_summary.json" \
  "$geometry_root/atlas_target_anchors.csv" \
  "$geometry_root/cell_target_geometry.csv" \
  "$geometry_root/target_pair_geometry.csv" \
  "$geometry_root/geometry_summary.json" \
  "$target_root/primary_16_target_calls.csv" \
  "$target_root/v3_gate_summary.json"; do
  test -f "$required"
  test ! -L "$required"
done

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" --version
"$python_bin" "$m1_projection_test"
"$python_bin" "$test_file"
"$python_bin" "$gate" \
  --contract "$contract" \
  --addendum "$addendum" \
  --inventory "$inventory" \
  --core-report "$core_report" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --target-root "$target_root" \
  --output-root "$output_root"
