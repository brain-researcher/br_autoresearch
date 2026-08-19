#!/bin/bash
#SBATCH --job-name=narps-fit
#SBATCH --partition=russpold
#SBATCH --account=russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-107%8
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/NARPS/outputs/logs/fit_%A_%a.out
#SBATCH --error=/oak/stanford/groups/russpold/users/zijiao/NARPS/outputs/logs/fit_%A_%a.err

set -euo pipefail

workspace=/oak/stanford/groups/russpold/users/zijiao/NARPS
output_root="$workspace/outputs"

cd "$workspace"
mkdir -p "$output_root/logs" "$output_root/tmp/joblib" \
  "$output_root/cache/matplotlib" "$output_root/cache/nilearn" \
  "$output_root/subject_maps"

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$output_root/tmp"
export JOBLIB_TEMP_FOLDER="$output_root/tmp/joblib"
export XDG_CACHE_HOME="$output_root/cache"
export MPLCONFIGDIR="$output_root/cache/matplotlib"
export NILEARN_DATA="$output_root/cache/nilearn"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load python/3.12.1

/home/users/zijiao/pcvenv/bin/python outputs/code/fit_subject.py \
  --subject-index "$SLURM_ARRAY_TASK_ID" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --chunk-voxels 12000
