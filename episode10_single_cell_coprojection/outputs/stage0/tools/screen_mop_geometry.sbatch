#!/usr/bin/env bash
#SBATCH --job-name=ep10-gao-geom
#SBATCH --partition=russpold,normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_geometry_v1/logs/%x-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_geometry_v1/logs/%x-%j.err

set -euo pipefail

base=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
python=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python

"$python" "$base/gao_lean_qualification/screen_mop_geometry.py" \
  --inventory "$base/gao_mop_soma_v1/mop_geometry_inventory.csv" \
  --output-csv "$base/gao_mop_geometry_v1/mop_geometry_support.csv" \
  --summary-json "$base/gao_mop_geometry_v1/mop_geometry_summary.json" \
  --milp-time-limit-seconds 60
