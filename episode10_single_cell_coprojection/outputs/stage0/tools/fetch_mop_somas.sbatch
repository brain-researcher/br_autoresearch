#!/usr/bin/env bash
#SBATCH --job-name=ep10-gao-soma
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_soma_v1/logs/%x-%j.out
#SBATCH --error=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/gao_mop_soma_v1/logs/%x-%j.err

set -euo pipefail

base=/scratch/users/zijiao/br_autoresearch/episode10_single_cell_coprojection
script="$base/gao_lean_qualification/fetch_mop_somas.py"
metadata="$base/gao_lean_qualification/mouse.neuron.info.json"
output="$base/gao_mop_soma_v1/mop_soma_inventory.csv"

ml purge
ml python/3.12.1
python3 "$script" \
  --metadata "$metadata" \
  --output "$output" \
  --workers 2
