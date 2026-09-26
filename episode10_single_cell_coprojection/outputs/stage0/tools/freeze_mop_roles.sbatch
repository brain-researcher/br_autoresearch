#!/usr/bin/env bash
#SBATCH --job-name=ep10-gao-role
#SBATCH --partition=russpold,normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_role_freeze_v1/logs/%x-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_role_freeze_v1/logs/%x-%j.err

set -euo pipefail

oak_episode=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
scratch_episode=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
python=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python

"$python" "$oak_episode/outputs/stage0/tools/freeze_mop_roles.py" \
  --inventory "$oak_episode/outputs/stage0/gao_mop_geometry_inventory.csv" \
  --output-csv "$scratch_episode/gao_mop_role_freeze_v1/gao_mop_frozen_roles.csv" \
  --summary-json "$scratch_episode/gao_mop_role_freeze_v1/gao_mop_role_freeze_summary.json" \
  --seed 20260926 \
  --milp-time-limit-seconds 300
