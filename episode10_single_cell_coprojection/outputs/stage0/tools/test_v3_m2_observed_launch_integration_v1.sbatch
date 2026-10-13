#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-obs-integ
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal
#SBATCH --output=/scratch/users/%u/br_autoresearch/episode10_single_cell_coprojection/%x-%j.out
#SBATCH --error=/scratch/users/%u/br_autoresearch/episode10_single_cell_coprojection/%x-%j.err

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
runtime_root="$scratch_root/v3_m2_observed_launch_integration_job_${SLURM_JOB_ID}"

test -x "$python_bin"
test -d "$project"
test ! -L "$project"
test -d "$scratch_root"
test ! -L "$scratch_root"
test ! -e "$runtime_root"
mkdir -m 2770 "$runtime_root"

shared_modules=(
  "$stage0/tools/fit_v3_m2_optional.py"
  "$stage0/tools/v3_m2_observed_session_v1.py"
  "$stage0/tools/v3_m2_selected_refit_v1.py"
  "$stage0/tools/v3_m2_trial_executor_v1.py"
  "$stage0/tools/v3_m2_m1_outer_import_v1.py"
  "$stage0/tools/load_v3_m2_observed_inputs_v1.py"
  "$stage0/tools/run_v3_m2_observed_search_v1.py"
  "$stage0/tools/launch_v3_m2_observed_search_v1.py"
)

corresponding_tests=(
  "$stage0/tests/test_fit_v3_m2_optional.py"
  "$stage0/tests/test_v3_m2_observed_session_v1.py"
  "$stage0/tests/test_v3_m2_selected_refit_v1.py"
  "$stage0/tests/test_v3_m2_trial_executor_v1.py"
  "$stage0/tests/test_v3_m2_m1_outer_import_v1.py"
  "$stage0/tests/test_load_v3_m2_observed_inputs_v1.py"
  "$stage0/tests/test_run_v3_m2_observed_search_v1.py"
  "$stage0/tests/test_launch_v3_m2_observed_search_v1.py"
)

run_tests=(
  "$stage0/tests/test_fit_v3_m2_optional.py"
  "$stage0/tests/test_v3_m2_trial_executor_v1.py"
  "$stage0/tests/test_v3_m2_observed_session_v1.py"
  "$stage0/tests/test_v3_m2_selected_refit_v1.py"
  "$stage0/tests/test_v3_m2_optional_adapter_v1.py"
  "$stage0/tests/test_v3_m2_permutation_production_adapter_v1.py"
  "$stage0/tests/test_v3_m2_rademacher_production_adapter_v1.py"
  "$stage0/tests/test_v3_m2_m1_outer_import_v1.py"
  "$stage0/tests/test_load_v3_m2_observed_inputs_v1.py"
  "$stage0/tests/test_run_v3_m2_observed_search_v1.py"
  "$stage0/tests/test_launch_v3_m2_observed_search_v1.py"
)

for required in "${shared_modules[@]}" "${corresponding_tests[@]}" "${run_tests[@]}"; do
  test -f "$required"
  test ! -L "$required"
done

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$runtime_root/pycache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "$project"

"$python_bin" -m py_compile "${shared_modules[@]}" "${corresponding_tests[@]}"

for test_file in "${run_tests[@]}"; do
  "$python_bin" "$test_file"
done

printf 'EP10 v3 M2 target-free observed-launch integration regression: PASS\n'
