#!/usr/bin/env bash
#SBATCH --job-name=ep10-mop-support
#SBATCH --partition=russpold,normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/%x-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/%x-%j.err

set -euo pipefail

oak_episode=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
scratch_episode=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
python=/scratch/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/ccf_env_py311/bin/python

"$python" "$oak_episode/outputs/stage0/tools/derive_mop_development_common_support.py" \
  --roles "$oak_episode/outputs/stage0/gao_mop_frozen_roles.csv" \
  --inventory "$oak_episode/outputs/stage0/gao_mop_geometry_inventory.csv" \
  --output "$scratch_episode/gao_mop_common_support_v1/gao_mop_development_assignment_support.csv" \
  --execute-safe-geometry-derivation
