#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m2-scale
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${OPTIONAL_QUALIFICATION_JOB_ID:?OPTIONAL_QUALIFICATION_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
tool="$stage0/tools/benchmark_v3_m2_search_scale_v1.py"
test_file="$stage0/tests/test_benchmark_v3_m2_search_scale_v1.py"
bridge_test="$stage0/tests/test_v3_m2_scale_benchmark_units_v1.py"
controller_test="$stage0/tests/test_v3_search_controller_v1.py"
executor_test="$stage0/tests/test_v3_m2_trial_executor_v1.py"
observed_session_test="$stage0/tests/test_v3_m2_observed_session_v1.py"
replay_test="$stage0/tests/test_v3_m2_benchmark_replay_v1.py"
optional_test="$stage0/tests/test_fit_v3_m2_optional.py"
diagnostic_test="$stage0/tests/test_v3_m2_optional_diagnostic_v1.py"
contract="$stage0/v3_m2_search_contract.yaml"
addendum="$stage0/v3_m2_numerical_falsifier_addendum_v1.yaml"
clarification="$stage0/v3_m2_numerical_falsifier_clarification_v2.yaml"
rank_clarification="$stage0/v3_m2_numerical_falsifier_clarification_v3.yaml"
inventory="$stage0/v3_m2_frozen_cell_inventory.tsv"

scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
geometry_root="$scratch_root/v3_m1_geometry_v1"
# This target root is passed only to the benchmark's sealed actual-K projection.
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"

core_report="$scratch_root/v3_m2_core_contract_qualification_job_45832824/v3_m2_core_contract_qualification.json"
actual_k_report="$scratch_root/v3_m2_actual_k_gate_v1_job_45847880/v3_m2_actual_k_gate_summary.json"
adequacy_report="$scratch_root/v3_m2_m1_adequacy_gate_job_45773980/v3_m2_m1_adequacy.json"
optional_report="$scratch_root/v3_m2_optional_family_qualification_job_${OPTIONAL_QUALIFICATION_JOB_ID}/v3_m2_optional_family_qualification.json"

output_root="$scratch_root/v3_m2_search_scale_benchmark_v1_job_${SLURM_JOB_ID}"
runtime_log="${output_root}.log"

test -x "$python_bin"
test -d "$scratch_root"
test ! -L "$scratch_root"
test ! -e "$output_root"
test ! -e "$runtime_log"

for required_root in "$interface_root" "$geometry_root" "$target_root"; do
  test -d "$required_root"
  test ! -L "$required_root"
done

for required in \
  "$tool" \
  "$test_file" \
  "$bridge_test" \
  "$controller_test" \
  "$executor_test" \
  "$observed_session_test" \
  "$replay_test" \
  "$optional_test" \
  "$diagnostic_test" \
  "$contract" \
  "$addendum" \
  "$clarification" \
  "$rank_clarification" \
  "$inventory" \
  "$core_report" \
  "$actual_k_report" \
  "$adequacy_report" \
  "$optional_report" \
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

exec >>"$runtime_log" 2>&1
printf 'EP10 v3 M2 target-free search-scale benchmark log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" --version
"$python_bin" "$bridge_test"
"$python_bin" "$test_file"
"$python_bin" "$controller_test"
"$python_bin" "$executor_test"
"$python_bin" "$observed_session_test"
"$python_bin" "$replay_test"
"$python_bin" "$optional_test"
"$python_bin" "$diagnostic_test"
"$python_bin" "$tool" \
  --contract "$contract" \
  --addendum "$addendum" \
  --hybrid-clarification "$clarification" \
  --hybrid-rank-clarification "$rank_clarification" \
  --inventory "$inventory" \
  --interface-root "$interface_root" \
  --geometry-root "$geometry_root" \
  --target-root "$target_root" \
  --core-report "$core_report" \
  --actual-k-report "$actual_k_report" \
  --adequacy-report "$adequacy_report" \
  --optional-report "$optional_report" \
  --output-root "$output_root"

printf 'EP10 v3 M2 target-free search-scale benchmark: PASS\n'
