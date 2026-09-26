#!/usr/bin/env bash
#SBATCH --job-name=ep10-stage0-preflight
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/stage0_preflight_v1/logs/%x-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/stage0_preflight_v1/logs/%x-%j.err

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"

preflight_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/stage0_preflight_v1"
input_dir="$preflight_root/input"
run_dir="$preflight_root/job_${SLURM_JOB_ID}"
python_bin="$SCRATCH/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python"

test -x "$python_bin"
test -d "$input_dir"
mkdir -p "$run_dir"

export EP10_SCREEN_MODULE="$input_dir/screen_mop_geometry.py"
export EP10_GEOMETRY_MODULE="$input_dir/build_whole_cortex_ssptr_geometry_support.py"

"$python_bin" -m py_compile \
  "$input_dir/build_mop_development_manifest.py" \
  "$input_dir/download_mop_development_swcs.py" \
  "$input_dir/derive_mop_development_common_support.py" \
  "$input_dir/construct_mop_terminal_targets.py" \
  "$input_dir/benchmark_fixed_k_normalizer.py" \
  "$input_dir/freeze_mop_roles.py" \
  "$input_dir/screen_mop_geometry.py" \
  "$input_dir/build_whole_cortex_ssptr_geometry_support.py"

"$python_bin" "$input_dir/construct_mop_terminal_targets.py" --self-test
"$python_bin" "$input_dir/test_construct_mop_terminal_targets.py"
"$python_bin" "$input_dir/benchmark_fixed_k_normalizer.py" --self-check

"$python_bin" "$input_dir/derive_mop_development_common_support.py" \
  --roles "$input_dir/gao_mop_frozen_roles.csv" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --output "$run_dir/gao_mop_development_assignment_support.csv" \
  --execute-safe-geometry-derivation
cmp "$run_dir/gao_mop_development_assignment_support.csv" \
  "$input_dir/gao_mop_development_assignment_support.csv"

"$python_bin" "$input_dir/build_mop_development_manifest.py" \
  --roles "$input_dir/gao_mop_frozen_roles.csv" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --common-support "$run_dir/gao_mop_development_assignment_support.csv" \
  --output "$run_dir/gao_mop_development_manifest.csv"
cmp "$run_dir/gao_mop_development_manifest.csv" \
  "$input_dir/gao_mop_development_manifest.csv"

"$python_bin" "$input_dir/download_mop_development_swcs.py" \
  --manifest "$run_dir/gao_mop_development_manifest.csv" \
  --roles "$input_dir/gao_mop_frozen_roles.csv" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --common-support "$run_dir/gao_mop_development_assignment_support.csv" \
  --output-root "$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1" \
  --validate-only

"$python_bin" "$input_dir/freeze_mop_roles.py" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --output-csv "$run_dir/gao_mop_frozen_roles.csv" \
  --summary-json "$run_dir/gao_mop_role_freeze_summary.json" \
  --seed 20260926 \
  --milp-time-limit-seconds 300
cmp "$run_dir/gao_mop_frozen_roles.csv" "$input_dir/gao_mop_frozen_roles.csv"
cmp "$run_dir/gao_mop_role_freeze_summary.json" \
  "$input_dir/gao_mop_role_freeze_summary.json"

"$python_bin" "$input_dir/benchmark_fixed_k_normalizer.py" \
  --output-json "$run_dir/exact_normalization_benchmark.json" \
  --primary-dimension 16 \
  --wide-dimension 56 \
  --wide-k 5 \
  --wide-state-cap 4000000 \
  --primary-state-cap 65536 \
  --maximum-dimension 64 \
  --coefficient-construction-cap-bytes 16777216 \
  --seed 20260926 \
  --timing-repetitions 99

printf 'EP10 stage-0 preflight complete: %s\n' "$run_dir"
