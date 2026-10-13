#!/usr/bin/env bash
#SBATCH --job-name=ep10-tag-profile
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1

project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
development="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/gao_mop_development_v1"

python3 "$project/outputs/stage0/tools/profile_mop_custom_tags.py" \
  --manifest "$project/outputs/stage0/gao_mop_development_manifest.csv" \
  --swc-root "$development/swc"
