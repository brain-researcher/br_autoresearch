#!/usr/bin/env bash
#SBATCH --job-name=ep10-mop-download
#SBATCH --partition=service
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
[[ "${SLURM_JOB_PARTITION:-}" == service ]]

input_dir="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_transfer_v1/input"
output_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1"
script="$input_dir/download_mop_development_swcs.py"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
python_bin="$(command -v python3)"
"$python_bin" --version

test -f "$script"

"$python_bin" "$script" \
  --manifest "$input_dir/gao_mop_development_manifest.csv" \
  --roles "$input_dir/gao_mop_frozen_roles.csv" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --common-support "$input_dir/gao_mop_development_assignment_support.csv" \
  --output-root "$output_root" \
  --validate-only

"$python_bin" "$script" \
  --manifest "$input_dir/gao_mop_development_manifest.csv" \
  --roles "$input_dir/gao_mop_frozen_roles.csv" \
  --inventory "$input_dir/gao_mop_geometry_inventory.csv" \
  --common-support "$input_dir/gao_mop_development_assignment_support.csv" \
  --output-root "$output_root" \
  --workers 4 \
  --execute-development-transfer
