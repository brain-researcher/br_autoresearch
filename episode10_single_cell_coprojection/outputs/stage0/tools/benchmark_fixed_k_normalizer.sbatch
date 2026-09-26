#!/usr/bin/env bash
#SBATCH --job-name=ep10-exactnorm
#SBATCH --partition=russpold,normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/users/%u/%x-%j.out
#SBATCH --error=/scratch/users/%u/%x-%j.err

# This short benchmark is intended to be aggregated with the other EP10
# pre-outcome checks. Do not submit it as a duplicate standalone job.
set -euo pipefail
umask 007

ml purge
ml load devel python/3.12.1

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"

oak_episode="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
scratch_episode="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
run_dir="$scratch_episode/exact_normalization_benchmark_v1/job_${SLURM_JOB_ID}"
python="$(command -v python3)"

mkdir -p "$run_dir"
cp "$oak_episode/outputs/stage0/tools/benchmark_fixed_k_normalizer.py" \
  "$run_dir/benchmark_fixed_k_normalizer.py"

"$python" --version

"$python" "$run_dir/benchmark_fixed_k_normalizer.py" \
  --output-json "$run_dir/exact_normalization_benchmark.json" \
  --primary-dimension 16 \
  --wide-dimension 56 \
  --wide-k 5 \
  --wide-state-cap 4000000 \
  --primary-state-cap 65536 \
  --maximum-dimension 64 \
  --coefficient-construction-cap-bytes 16777216 \
  --seed 20260926
